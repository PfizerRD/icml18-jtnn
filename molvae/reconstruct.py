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
from jtnn import evalutils

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-f", "--test_files", dest="test_files", help='comma-separated list of file names')
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()

   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

files = opts.test_files.split(',')

for file in files:
    evalutils.print_reconstruction_accuracy(os.path.join(opts.test_path, file), model)
    

