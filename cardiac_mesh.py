import glob
import os
import numpy as np
import random
import scipy
import re

# I rename the class VTKObject as Mesh so I don't have to change the function calls in the code
from VTK.VTKMesh import VTKObject as Mesh  #, MeshViewer, MeshViewers
import meshio
import time
from copy import deepcopy
import random
from sklearn.decomposition import PCA
from tqdm import tqdm # for the progress bar
import scipy

class CardiacMesh(object):
    
    subparts_ids = {
      "all": None,
      "LV": [1, 2],
      "LV_endo": [1],
      "LV_epi": [2],
      "LA": [3],
      "RV": [4],
      "RA": [5],  
    } 

    def __init__(self, nVal, nTraining, train_file, reference_mesh_file, ids_file=None, pca_n_comp=8, fitpca=False, subpart=None):

        self.nVal = nVal
        self.nTraining = nTraining
        self.train_file = train_file
        self.vertices_train, self.vertices_val, self.vertices_test = None, None, None
        self.ids_file = ids_file

        self.N = None
        self.n_vertex = None
        self.fitpca = fitpca

        self.reference_mesh = Mesh(filename=reference_mesh_file) # to extract adjacency matrix
        if subpart is not None:
            self.reference_mesh = self.reference_mesh.extractSubpart(subpart)
        
        self.load()

        self.mean = np.mean(self.vertices_train, axis=0)
        self.std = np.std(self.vertices_train, axis=0)
        self.pca = PCA(n_components=pca_n_comp)
        self.pcaMatrix = None
        self.normalize()


    def load(self):
        '''
        Load numpy data files containing train and test files
        :return:
        '''

        ## In spite of its name, the self.train_file should contain all the data
        ## TODO: change this attribute's name
        all_vertices = np.load(self.train_file, allow_pickle=True   ) # training + validation + testing

        all_vertices = np.array([
            scipy.spatial.procrustes(
                all_vertices[i],
                self.reference_mesh.points)[0]
            for i in range(len(all_vertices))
        ])

        # Mean and std. deviation are computed based on all the samples (not only the training ones)
        self.mean, self.std = np.mean(all_vertices, axis=0), np.std(all_vertices, axis=0)

        ids = [x.strip() for x in open(self.ids_file)]

        self.train_ids = ids[:self.nTraining]
        self.val_ids = ids[self.nTraining: (self.nTraining + self.nVal)]
        self.test_ids = ids[(self.nTraining + self.nVal):] 

        self.vertices_train = all_vertices[:self.nTraining]
        self.vertices_val = all_vertices[self.nTraining:(self.nTraining+self.nVal)]
        self.vertices_test = all_vertices[(self.nTraining+self.nVal):]

        self.n_vertex = self.vertices_train.shape[1]


    def normalize(self):

        self.vertices_train = (self.vertices_train - self.mean) / self.std
        self.vertices_val = (self.vertices_val - self.mean) / self.std
        self.vertices_test = (self.vertices_test - self.mean) / self.std

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

        print("Train data of size: {}\nTest data of size: {} ".format(len(self.datapaths["train"]), len(self.datapaths["test"])))


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

    

class NumpyFromVTKs(object):
    
    """docstring for NumpyFromVTKs"""

    def __init__(self, folders, 
                   filename_pattern, 
                   dataset_name, 
                   partition_ids=None, 
                   subj_ids=None, N_subj=None):
        
        '''          
          - subj_ids: list of subject IDs to draw samples from (if None, it's all the subjects)
          - N_subj: number of subjects to sample (if None, it's all the subjects from subj_ids)
        '''
        
        self.folders = folders if isinstance(folders, list) else [folders]
        self.filename_pattern = os.path.join(folders, filename_pattern)
        self.regex = re.compile(self.filename_pattern.replace("*","(.*)"))
        
        self.dataset_name = dataset_name # output directory        
        # self.output_filename = os.path.join(self.dataset_name, output_filename)
        # self.ids_filename = self.output_filename + "_subj_ids.txt"
        
        self.partition_ids = partition_ids
        self.N_subj = N_subj
        self.subj_ids = subj_ids 

        self.gather_paths()
        self.vertices = self.gather_data()
       
           
    def extract_id_from_path(self, path):
        id = self.regex.match(path).group(1)
        return id
        
        
    def gather_paths(self):
        
        '''
          Gather all the paths associated to mesh files
        '''

        datapaths = []
        for subdir_name in self.folders:
            datapaths.extend(sorted(glob.glob(self.filename_pattern)))
        
        path_dict = {self.extract_id_from_path(x): x for x in datapaths}
            
        if self.subj_ids is None:
            ids = path_dict.keys() # [extract_id_from_path(x) x for x in datapaths]
        else:
            ids = [x for x in path_dict.keys() if x in subj_ids]
                                        
        if self.N_subj is not None:
            random.shuffle(ids)            
            # TODO: add sanity check for case when N_subj > len(ids)
            ids = ids[0:self.N_subj]        
            
        datapaths = [path_dict[i] for i in ids]
        self.subj_ids = ids
        self.datapaths = datapaths


    def gather_data(self):
        vertices = []

        # tqdm: for progress bar (I think)
        for p in tqdm(self.datapaths, unit="subjects"):
            mesh_filename = p
            mesh = Mesh(filename=mesh_filename, load_connectivity=False) # Load mesh
            mesh = Mesh.extractSubpart(mesh, self.partition_ids)
            vertices.append(mesh.points)

        return np.array(vertices)


    def save_vertices(self, output_filename, ids_filename=None):

        #print(output_filename)
        #print(os.path.dirname(output_filename))

        if not os.path.exists(os.path.dirname(output_filename)):
            print(os.path.dirname(output_filename))
            os.makedirs(os.path.dirname(output_filename))

        if not output_filename.endswith(".npy"):
            output_filename += ".npy"

        if ids_filename is None:
            ids_filename = output_filename if not output_filename.endswith(".npy") else output_filename[:-4]
            ids_filename += "_subject_ids.txt"
            #print(ids_filename)

        if not os.path.exists(output_filename) and not os.path.exists(ids_filename):
            np.save(output_filename, self.vertices)
            with open(ids_filename, "w") as ff:
                ff.write("\n".join(self.subj_ids))
        else:
            print("File already exists")
            return 1
        
        return 0
    
    
def generateNumpyDataset(data_path, save_path, partition, subj_ids=None, N_subj=None):
    # folders = "/MULTIX/DATA/INPUT/disk_2/coma/Cardio/meshes/vtk_meshes",
    # dataset_name="/MULTIX/DATA/INPUT/disk_2/coma/Cardio/meshes/LV_all_subjects",
    npy = NumpyFromVTKs(
      folders=data_path,
      filename_pattern="*/output.001.vtk",
      dataset_name=save_path,
      subj_ids=subj_ids,
      N_subj=N_subj, # all
      partition_ids=CardiacMesh.subparts_ids[partition]
    )

    npy.save_vertices(save_path)
    
    return 0
