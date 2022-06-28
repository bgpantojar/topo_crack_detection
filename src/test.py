#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:46:06 2020

This script contains the codes to segment cracks.
The codes are based on "TOPO-Loss for continuity-preserving crack detection using deep learning" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.conbuildmat.2022.128264

@author: pantoja
"""


import os
import numpy as np
import cv2 as cv
import torch
from math import sqrt as sqrt
from data_set import open_dataset
from network import UNet16
from torch.utils.data import DataLoader
from sliding_window import main_sliding
from image_from_slices import main_image_from_slices

# model path for different experiments
#model_path = '../models/mse/mse_full_weights_50_5e-06_1.0_0.1.pt' #mse
#model_path = '../models/topo/topo_full_weights_50_3e-05_100.0_10.0_last.pt' #topo
#model_path = '../models/dice+topo/dice+topo_full_weights_50_3e-05_100.0_10.0_last.pt' #dice+topo
model_path = '../models/mse+topo/mse+topo_full_weights_50_3e-05_100.0_10.0.pt' #mse+topo

#Data paths
path2test="../data/test_set/images_patches"
full_image_path = "../data/test_set/images/"
patches_path="../results/patches/"

#Creating patches_images
desired_size = 1024 #if overflow the memory, reduce the size
main_sliding('test_set','images', desired_size=desired_size)

test_ds=open_dataset(path2test, transform='test')   
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
images_path = path2test + '/'
# load a trained model
model = UNet16()
device = torch.device('cpu')
model=model.to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.train(False)
model.eval()

#Dictionaries to storage prediction and images information    
prediction_bin = {}
prediction_rgb = {}
prediction_dm = {}

for ni, image in enumerate(test_dl):
    image = image.to(device)
    image_size = image.size()
    print(image_size)
    SR = model(image)
    SR_probs = SR
    SR_probs_arr = SR_probs.detach().numpy().reshape(image_size[2], image_size[3])

    #Threshold defined according experiments
    #binary_result = SR_probs_arr > 6 #threshold for mse
    #binary_result = SR_probs_arr > 2 #threshold for topo
    binary_result = SR_probs_arr > 2 #threshold for dice+topo
    #binary_result = SR_probs_arr > 4 #threshold for mse+topo

    image_numpy = image.detach().numpy()
    image_numpy = image_numpy[0, 0, :, :]
    image_name = test_dl.dataset.path2imgs[ni].split('/')
    image_name = image_name[-1].split(".")
    image_name = image_name[0]

    prediction_dm[image_name] = SR_probs_arr

    corner = np.array(binary_result, dtype='uint8')
    prediction_bin[image_name] = (1-corner)*255
    
    cshp = corner.shape
    corner_rgb = np.zeros((cshp[0], cshp[1], 3), dtype='uint8')
    corner_rgb[:,:,2] = (1-corner)*255
    prediction_rgb[image_name] = corner_rgb

#Getting original images
list_images = os.listdir(images_path)
images = {}
for img in list_images:
    #print(images_path + img)
    images[img[:-4]] = cv.imread(images_path + img)


#Resizing predictions to the original size
prediction_bin_resized = {}
prediction_rgb_resized = {}
for key in images:
    prediction_bin_resized[key] = cv.resize(prediction_bin[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)
    prediction_rgb_resized[key] = cv.resize(prediction_rgb[key], (images[key].shape[1],images[key].shape[0]), interpolation = cv.INTER_CUBIC)

#Overlaying prediction_rgb with original image
overlayed_prediction = {}
for key in images:
    overlayed_prediction[key] = cv.addWeighted(images[key], 1.0, prediction_rgb_resized[key], 0.7, 0)

#saving binary and overlayed predictions
#Check if directory exists, if not, create it
check_dir = os.path.isdir('../results/patches/')
if not check_dir:
    os.makedirs('../results/patches/')

for key in images:
    cv.imwrite('../results/patches/' + key + '_pred_bin.png', prediction_bin_resized[key])
    cv.imwrite('../results/patches/' + key + '.png', images[key])
    cv.imwrite('../results/patches/' + key + '_overlayed.png', overlayed_prediction[key])
    cv.imwrite('../results/patches/' + key + '_pred_dm.png', 5*prediction_dm[key])

main_image_from_slices(full_image_path, patches_path, type_img = None, windowSize=(desired_size,desired_size))
main_image_from_slices(full_image_path, patches_path, type_img = "pred_bin", windowSize=(desired_size,desired_size))
main_image_from_slices(full_image_path, patches_path, type_img = "overlayed", windowSize=(desired_size,desired_size))
main_image_from_slices(full_image_path, patches_path, type_img = "pred_dm", windowSize=(desired_size,desired_size))