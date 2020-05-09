from __future__ import print_function
from lib import models, graph, coarsening, utils, mesh_sampling
from lib.visualize_latent_space import visualize_latent_space
import numpy as np
import json
import os
import copy
from cardiac_mesh import CardiacMesh
from opendr.topology import get_vert_connectivity
import time
import yaml


def generate_model(params_, add_hash=True):

  params = copy.deepcopy(params_)
  params['nz'] = [params['nz']]

  print("Loading data .. ")

  np.random.seed(params.get('random_seed', 0))
  
  # train_file = os.path.join(params['data_dir'], 'train.npy') if params.get('train_file', None) is None else params['train_file'].format(config.data_dir)
  # test_file = os.path.join(params['data_dir'], 'test.npy') if params.get('test_file', None) is None else params['test_file'].format(config.data_dir)
  train_file = params['data_file']

  try:
      nVal = min(params['nVal_fraction'] * params['nTraining'], params['nVal_max'])
  except:
      nVal = params['nVal']
 
  # Read cardiac meshes from npy files  
  cardiac_data = CardiacMesh(
      nTraining = params['nTraining'],
      nVal = nVal,
      train_file = train_file,
      ids_file = params['ids_file'],
      reference_mesh_file = params['reference_mesh_file'],
      pca_n_comp = params['nz'],
      subpart = CardiacMesh.subparts_ids[params['partition']]
  )  
 
  # cardiac_data = CardiacMesh(
  #    nVal = 100,
  #    nTraining = 2500,
  #    train_file = train_file,
  #    test_file = test_file,
  #    reference_mesh_file = params['reference_mesh_file'],
  #    pca_n_comp = params['nz'],
  #    subpart = subparts_ids[params['partition']]
  #)
  
  print("Generating Transform Matrices ..")
  # Generates adjacency matrices A, downsampling matrices D, and upsampling matrices U by sampling
  # the mesh 4 times. Each time the mesh is sampled by a factor of 4
  M, A, D, U = mesh_sampling.generate_transform_matrices(cardiac_data.reference_mesh, params['ds_factors'])
  A, D, U = tuple( [ map(lambda x: x.astype('float32'), kk) for kk in (A, D, U) ] )
  
  print("Computing Graph Laplacians ..")
  L = [graph.laplacian(a, normalized=True) for a in A]
  
  # 
  X_train = cardiac_data.vertices_train.astype('float32')
  X_val = cardiac_data.vertices_val.astype('float32')
  X_test = cardiac_data.vertices_test.astype('float32')
  
  n_train = X_train.shape[0]
  
  params['nv']             = cardiac_data.n_vertex  # Number of vertices in the input mesh
  params['F_0']            = int(X_train.shape[2])  # Number of graph input features.
  params['p']              = map(lambda x: x.shape[0], A)    # Pooling sizes.
  params['decay_steps']    = n_train / params['batch_size']
 
  del params['random_seed']
  del params['reference_mesh_file']
  del params['ds_factors']
  del params['data_file']
  del params['ids_file']
  del params['partition']
  del params['nTraining']
  del params['nVal']
  del params['nVal_max']
  del params['nVal_fraction']
  del params['nTest']
  del params['hash']

  params['regularization'] = float(params['regularization'])
  params['learning_rate'] = float(params['learning_rate'])

  if add_hash:  
    # this computes a hash and adds it to the file name
    # which allows to link biunivocally a given file of parameters and a run of the algorithm
    import datetime  
    timestamp = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
    params['dir_name'] = params['dir_name'] + "__{}".format(timestamp)

  model = models.coma(L=L, D=D, U=U, **params)

  model.reference_mesh = cardiac_data.reference_mesh
  model.std = cardiac_data.std
  model.mean = cardiac_data.mean

  return model


def run(params):

  checkpoints_dir = params["checkpoint_dir"]
  
  model = generate_model(params)
  
  model_dirname = os.path.join(model.checkpoint_dir, model.dir_name)

  if not os.path.exists(model_dirname):

    from cardiac_mesh import CardiacMesh

    # Read cardiac meshes from npy files
    # train_file = os.path.join(params['data_dir'], 'train.npy') if params.get('train_file', None) is None else params['train_file'].format(params['data_dir'])
    # test_file = os.path.join(params['data_dir'], 'test.npy') if params.get('test_file', None) is None else params['test_file'].format(params['data_dir'])
    train_file = params['data_file']

    try:
        nVal = min(params['nVal_fraction'] * params['nTraining'], params['nVal_max']),
    except:
        nVal = params['nVal']
 
    cardiac_data = CardiacMesh(
        nTraining = params['nTraining'],
        nVal = min(params['nVal_fraction'] * params['nTraining'], params['nVal_max']),
        train_file = train_file,
        ids_file = params['ids_file'],
        reference_mesh_file = params['reference_mesh_file'],
        pca_n_comp = params['nz'],
        subpart = CardiacMesh.subparts_ids[params['partition']]
    )  

    X_train = cardiac_data.vertices_train.astype('float32')
    X_val = cardiac_data.vertices_val.astype('float32')
    X_test = cardiac_data.vertices_test.astype('float32')
  
    n_train = X_train.shape[0]

    timestamp = model_dirname.split("__")[-1]
    train_id_file = os.path.join(os.path.dirname(model_dirname), timestamp + "_training_ids.txt")
    test_id_file  = os.path.join(os.path.dirname(model_dirname), timestamp + "_testing_ids.txt")
    val_id_file   = os.path.join(os.path.dirname(model_dirname), timestamp + "_validation_ids.txt")

    open(train_id_file, "w").write("\n".join([str(x) for x in cardiac_data.train_ids]))
    open(test_id_file, "w").write("\n".join([str(x) for x in cardiac_data.test_ids]))
    open(val_id_file, "w").write("\n".join([str(x) for x in cardiac_data.val_ids]))

    # if not os.path.exists(os.path.join(checkpoints_dir, params['dir_name'])):
    os.makedirs(model_dirname)
    with open(model_dirname + "_params.json",'w') as fp:
      saveparams = copy.deepcopy(params)
      saveparams['seed'] = params.get('random_seed', 0)
      json.dump(saveparams, fp)
    
    loss, t_step = model.fit(X_train, X_train, X_val, X_val)

  else:
    print("Folder {} already exists.".format(model_dirname))


if __name__ == "__main__":
  
  import argparse

  parser = argparse.ArgumentParser(description='Tensorflow Trainer for Convolutional Mesh Autoencoders')
  parser.add_argument('--config_yaml', default='config.yaml', help='YAML configuration file ')
  parser.add_argument('--device_id', default='0', help='GPU ID')

  args = parser.parse_args()

  params = yaml.safe_load(open(args.config_yaml))

  os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

  run(params)
