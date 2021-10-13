# Created by  IT-JIM  2021
# Here I fool around with LeCun's transformer
# And try to solve the issues with torchtext version incompatibilities

import sys

import torch
import torchtext


########################################################################################################################
def main():
    dset_train, dset_val = torchtext.datasets.IMDB(root='/home/seymour/data')
    print(type(dset_train), len(dset_train))
    print(type(dset_val), len(dset_val))

########################################################################################################################
if __name__ == '__main__':
    main()