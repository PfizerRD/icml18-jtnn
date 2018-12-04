import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
import os
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem

from jtnn import *


def print_reconstruction_accuracy(file, model):
    print(file)

    data = []
    try:
        with open(file) as f:
            for line in f:
                s = line.strip("\r\n ").split()[0]
                data.append(s)
    except:
        print('Could not open {}'.format(file))
        return None

    # remove duplicates
    data = list(set(list(data)))

    acc, tot = 0, 0
    acc_canon, tot_canon = 0,0
    print("Total test data: {}".format(len(data)))
    for smiles in data:
        try:
            mol = Chem.MolFromSmiles(smiles)
            smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
            dec_smiles = model.reconstruct(smiles3D)
        except:
            continue

        if dec_smiles == smiles3D:
            acc += 1

        tot += 1
        tot_canon += 1

        try:
            original_canon = chemutils.canonicalize_smiles(smiles3D)
            new_canon = chemutils.canonicalize_smiles(dec_smiles)
        except:
            continue

        if original_canon == new_canon:
            acc_canon += 1

        """
        dec_smiles = model.recon_eval(smiles3D)
        tot += len(dec_smiles)
        for s in dec_smiles:
            if s == smiles3D:
                acc += 1
        print acc / tot
        """

    print("reconstructed {} / {}, accuracy: {}".format(acc, tot, float(acc)/tot))
    print("canon reconstructed {} / {}, accuracy: {}".format(acc_canon, tot_canon, float(acc_canon)/tot_canon))
    
