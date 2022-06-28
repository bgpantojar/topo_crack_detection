#!/usr/bin/env python

"""
This script contains the codes to segment cracks.
The codes are based on "TOPO-Loss for continuity-preserving crack detection using deep learning" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.conbuildmat.2022.128264

This script specifically support deep learning codes development.
These are based on codes published in:
"Avendi, M., 2020. PyTorch Computer Vision Cookbook:
Over 70 Recipes to Master the Art of Computer Vision with Deep Learning and PyTorch 1. x. Packt Publishing Limited."

Slightly changes are introduced to addapt to general pipeline

@author: pantoja
"""

# import necessary modules
import random
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from data_set import open_dataset
from network import UNet16
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import optim
from loss_opt import *
from train import *
from torch import nn
from newloss import MALISLoss_window_pos
import argparse

#Setting input hyperparameters with argparse
parser = argparse.ArgumentParser(description='TOPO Crack Detection')
parser.add_argument('--lr', action='store', default=0.00003, help='Learning rate')
parser.add_argument('--n_epoch', action='store', default = 50, help='Number of epoch')
parser.add_argument('--malis_neg', action='store', default = 100, help='Negative Malis parameter')
parser.add_argument('--malis_pos', action='store', default = 10, help='Positive Malis parameter')
parser.add_argument('--model_name', action='store', default = 'mse+topo', help='Model loss required. mse+topo, mse, topo, dice+topo')
args = parser.parse_args()

#Hyperparameters
lr = float(args.lr)
n_epoch = int(args.n_epoch)
malis_neg = float(args.malis_neg)
malis_pos = float(args.malis_pos)
model_name = str(args.model_name)

# To have reproducible results the random seed is set to 42.
seed = 27
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Paths to images and their masks
#train
path2train_i="../data/training_set/c_images"
path2train_m="../data/training_set/c_mask"
#validation
path2valid_i="../data/validation_set/c_images"
path2valid_m="../data/validation_set/c_mask"

##DATA LOADER
#Defining two objects of open_dataset class:
train_ds=open_dataset(path2train_i, path2data_m=path2train_m, transform='train')
valid_ds=open_dataset(path2valid_i, path2data_m=path2valid_m, transform='val')

#Create Data Loader
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(valid_ds, batch_size=16, shuffle=False) 


##MODEL
#Define an object of network class
model = UNet16(pretrained=True)

#Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=model.to(device)
#Print the mode
print(model)
# Show the model summary
summary(model, input_size=(3, 256, 256))

#Hyperparameters
##Optimizer
opt = optim.Adam(model.parameters(), lr=lr)
loss_func1 = nn.MSELoss()
loss_func2 = MALISLoss_window_pos()

##Trainning Model
path2models= "../models/"
params_train={
    "num_epochs": n_epoch,
    "optimizer": opt,
    "loss_func1": loss_func1, #mse
    "loss_func2": loss_func2, #malis - topo
    "loss_func": loss_func, #dice
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "path2weights": path2models+model_name+"_full_weights_{}_{}_{}_{}.pt".format(n_epoch, lr, malis_neg, malis_pos),
    "path2losshist": path2models+model_name+"_loss_history_{}_{}_{}_{}.json".format(n_epoch, lr, malis_neg, malis_pos),
    "path2metrichist": path2models+model_name+"_metric_history_{}_{}_{}_{}.json".format(n_epoch, lr, malis_neg, malis_pos),
    "malis_params": [malis_neg, malis_pos],
    "model_name" : model_name
}
#Train
model, loss_hist, metric_hist = train_val(model,params_train)