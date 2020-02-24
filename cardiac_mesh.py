import glob
import os
import numpy as np
import random

# I rename the class VTKObject as Mesh so I don't have to change the function calls in the code
from VTK.VTKMesh import VTKObject as Mesh  #, MeshViewer, MeshViewers
import meshio
import time
from copy import deepcopy
import random
from sklearn.decomposition import PCA
from tqdm import tqdm # for the progress bar

class CardiacMesh(object):

    def __init__(self, nVal, train_file, test_file, reference_mesh_file, pca_n_comp=8, fitpca=False):

        self.nVal = nVal
        self.train_file, self.test_file = train_file, test_file
        self.vertices_train, self.vertices_val, self.vertices_test = None, None, None

        self.N = None
        self.n_vertex = None
        self.fitpca = fitpca

        self.mean, self.std = None, None

        self.load()

        self.reference_mesh = Mesh(filename=reference_mesh_file) # to extract adjacency matrix
        self.reference_mesh = self.reference_mesh.extractSubpart([1,2])

        # self.mean = np.mean(self.vertices_train, axis=0)
        # self.std = np.std(self.vertices_train, axis=0)
        self.pca = PCA(n_components=pca_n_comp)
        self.pcaMatrix = None
        self.normalize()


    def load(self):
        '''
        Load numpy data files containing train and test files
        :return:
        '''

        vertices_train = np.load(self.train_file)
        self.mean, self.std = np.mean(vertices_train, axis=0), np.std(vertices_train, axis=0)

        self.vertices_train = vertices_train[:-self.nVal]
        self.vertices_val = vertices_train[-self.nVal:]

        self.n_vertex = self.vertices_train.shape[1]

        self.vertices_test = np.load(self.test_file)


    def normalize(self):

        def normalize_(x):
            x = x - x.mean()
            x = x / x.std()
            return x

        self.vertices_train = normalize_(self.vertices_train)
        self.vertices_val = normalize_(self.vertices_val)
        self.vertices_test = normalize_(self.vertices_test)

        self.N = self.vertices_train.shape[0]

        if self.fitpca:
            self.pca.fit(np.reshape(self.vertices_train, (self.N, self.n_vertex*3) ))

        # eigenVals = np.sqrt(self.pca.explained_variance_)
        # self.pcaMatrix = np.dot(np.diag(eigenVals), self.pca.components_)
        print('Vertices normalized')


    def vec2mesh(self, vec):
        vec = vec.reshape((self.n_vertex, 3))*self.std + self.mean
        return Mesh(v=vec, f=self.reference_mesh.f)
    def show(self, ids):
        '''ids: list of ids to play '''
        if max(ids)>=self.N:
            raise ValueError('id: out of bounds')

        mesh = Mesh(v=self.vertices_train[ids[0]], f=self.reference_mesh.f)
        time.sleep(0.5)    # pause 0.5 seconds
        viewer = mesh.show()
        for i in range(len(ids)-1):
            viewer.dynamic_meshes = [Mesh(v=self.vertices_train[ids[i+1]], f=self.reference_mesh.f)]
            time.sleep(0.5)    # pause 0.5 seconds
        return 0
    def sample(self, BATCH_SIZE=64):
        datasamples = np.zeros((BATCH_SIZE, self.vertices_train.shape[1]*self.vertices_train.shape[2]))
        for i in range(BATCH_SIZE):
            _randid = random.randint(0,self.N-1)
            #print _randid
            datasamples[i] = ((deepcopy(self.vertices_train[_randid]) - self.mean)/self.std).reshape(-1)

        return datasamples

    def save_meshes(self, filename, meshes):
        for i in range(meshes.shape[0]):
            vertices = meshes[i].reshape((self.n_vertex, 3)) * self.std + self.mean
            mesh = Mesh(v=vertices, f=self.reference_mesh.f)
            # TODO: replace the write function
            # mesh.write_ply(filename+'-'+str(i).zfill(3)+'.ply')
        return 0

    def show_mesh(self, viewer, mesh_vecs, figsize):
        for i in range(figsize[0]):
            for j in range(figsize[1]):
                mesh_vec = mesh_vecs[i*(figsize[0]-1) + j]
                mesh_mesh = self.vec2mesh(mesh_vec)
                viewer[i][j].set_dynamic_meshes([mesh_mesh])
        time.sleep(0.1)    # pause 0.5 seconds
        return 0

    def get_normalized_meshes(self, mesh_paths):
        meshes = []
        for mesh_path in mesh_paths:
            mesh = Mesh(filename=mesh_path)
            mesh_v = (mesh.v - self.mean)/self.std
        meshes.append(mesh_v)
        return np.array(meshes)


class MakeSlicedTimeDataset(object):
    """docstring for FaceMesh"""

    def __init__(self, folders, folder_structure="*.vtk", dataset_name="LV", partition_ids=None, N_subj=None):

        self.folders = folders if isinstance(folders, list) else [folders]
        self.folder_structure = folder_structure
        self.partition_ids = partition_ids
        self.dataset_name = dataset_name
        self.N_subj = N_subj

        self.gather_paths()
        self.train_vertices = self.gather_data(self.datapaths["train"])
        self.test_vertices = self.gather_data(self.datapaths["test"])

        self.save_vertices()


    def gather_paths(self, test_fraction=0.1):
        '''
        :param opt: 'train' or 'test'
        :return:
        '''

        datapaths = []
        for subdir_name in self.folders:
            datapaths.extend(sorted(glob.glob(os.path.join(subdir_name, self.folder_structure))))

        if self.N_subj is not None:
            datapaths = datapaths[1:self.N_subj]

        train_indices = list(range(len(datapaths)))

        random.shuffle(train_indices)
        train_indices = train_indices[0:int((1-test_fraction)*len(datapaths))]
        test_indices = [i for i in range(0, len(datapaths)) if i not in train_indices]
        self.datapaths = {}
        self.datapaths["train"] = [datapaths[i] for i in train_indices]
        self.datapaths["test"] = [datapaths[i] for i in test_indices]

        print("Train data of size: {}\nTest data of size: {} ".format(len(self.datapaths["train"]), len(self.datapaths["train"])))


    def gather_data(self, datapaths):
        vertices = []

        # tqdm: for progress bar (I think)
        for p in tqdm(datapaths, unit="subjects"):
            mesh_filename = p
            mesh = Mesh(filename=mesh_filename) # Load mesh
            mesh = Mesh.extractSubpart(mesh, self.partition_ids)
            vertices.append(mesh.points)

        return np.array(vertices)


    def save_vertices(self):

        if not os.path.exists(self.dataset_name):
            os.makedirs(self.dataset_name)

        train_folder = os.path.join(self.dataset_name, 'train')
        test_folder = os.path.join(self.dataset_name, 'test')

        np.save(train_folder, self.train_vertices)
        np.save(test_folder, self.test_vertices)

        print("Saving ... {}".format(train_folder))
        print("Saving ... {}".format(test_folder))

        return 0


def generateSlicedTimeDataSet(data_path, save_path):

    MakeSlicedTimeDataset(folders=folder, folder_structure="*/*vtk", partition_ids=[1, 2], N_subj=5000)
    # MakeSlicedTimeDataset(folders=[data_path], dataset_name=os.path.join(save_path, 'sliced'))
    return 0
