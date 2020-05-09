import argparse
import os
from cardiac_mesh import *

parser = argparse.ArgumentParser(
    description='Preprocessing data for Convolutional Mesh Autoencoders',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--data', dest='data', type=str, required=True,
    help='path to the data directory'
)

parser.add_argument(
    '--save_path', dest='save_path', type=str, default='data',
    help='path where processed data will be saved'
)

parser.add_argument(
    '--partition', dest='partition', type=str,
    help='partition of the mesh, i.e. chamber(s) of the heart'
)

parser.add_argument(
    '--N_subj', dest='N_subj', type=int,
    help='Number of subjects to store into the npy object'
)

def main():

    args = parser.parse_args()

    #if not os.path.exists(args.save_path):
        #os.makedirs(args.save_path)

    print("Preprocessing data")

    # TODO: Change name!!!
    generateNumpyDataset(args.data, args.save_path, args.partition, N_subj=args.N_subj)


if __name__ == '__main__':
    main()
