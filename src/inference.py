#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:10:22 2021

@author: pantoja
"""

import os
import numpy as np
import cv2 as cv
import torch
from network import UNet16
from tqdm import tqdm
import skimage.io
from sliding_window import zero_pad
from sliding_window import sliding_window
import glob
import torchvision.transforms as T
import warnings


def crack_segmentation(images_path, out_return=False):

    model_path = '../models/mse+topo_full_weights_200_0.00018_0.01_0.001.pt' #mse+topo
    warnings.filterwarnings("ignore")
    result_path="../results/"

    model = UNet16()
    device = torch.device('cuda')
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.to(device)
    model.train(False)
    model.eval()

    desired_size = 256#1024#512
    transform = []
    transform.append(T.Resize((desired_size, desired_size)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    transform = T.Compose(transform)

    image_names=[i.split(os.sep)[-1] for i in glob.glob(os.path.join(images_path, '*.png'))]
    image_names+=[i.split(os.sep)[-1] for i in glob.glob(os.path.join(images_path, '*.JPG'))]
    image_names+=[i.split(os.sep)[-1] for i in glob.glob(os.path.join(images_path, '*.jpg'))]

    #Dictionary to save binary predictions
    crack_prediction_bin = {}

    print("Starting crack prediction----")
    for image_name in tqdm(image_names):
        key = image_name.split('.')[0]
        image_file = skimage.io.imread(os.path.join(images_path, image_name))
        org_im_h = image_file.shape[0]
        org_im_w  = image_file.shape[1]
        padded_image = zero_pad(image_file, desired_size)

        window_names = []
        windows = [] # as Tensor (ready for to use for deep learning)
        for (x, y, window) in sliding_window(padded_image, stepSize=desired_size, windowSize=(desired_size, desired_size)):
                window_names.append(image_name[:-4]+  "_{:d}".format(x) + "_{:d}".format(y))
                window = T.ToPILImage()(window) # as PIL
                window = transform(window)
                windows.append(torch.reshape(window, [1, 3, desired_size, desired_size]))


        overlay_crack = zero_pad(np.zeros((org_im_h, org_im_w), dtype = "uint8"),
                            desired_size = desired_size)

        with torch.no_grad():
            for window, window_name in zip(windows, window_names):
                window = window.to(device)
                SR = model(window)
                #SR_probs = torch.sigmoid(SR)
                SR_probs = SR
                SR_probs_arr = SR_probs.view(desired_size,desired_size)
                #SR_probs.detach().numpy().reshape(desired_size, desired_size)
                #binary_result = SR_probs_arr > 0.5
                binary_result = SR_probs_arr < 2
                binary_result = binary_result.to('cpu').detach().numpy()
                y = int(window_name.split('_')[-1])
                x = int(window_name.split('_')[-2])
                overlay_crack[y:y + desired_size, x:x + desired_size] = binary_result
        #overlay_crack = overlay_crack[:org_im_h, :org_im_w] * 255
        overlay_crack = overlay_crack[:org_im_h, :org_im_w] * 255

        image_name_save = 'crack_' + key + '_mask.png'
        skimage.io.imsave(result_path + image_name_save, overlay_crack)
        image_name_save = 'crack_' + key + '_photo.jpg'
        skimage.io.imsave(result_path + image_name_save, image_file)

        overlay_name_save = 'crack_' + key + '_overlay.jpg'
        prediction_rgb = np.zeros((overlay_crack.shape[0], overlay_crack.shape[1], 3), dtype='uint8')
        prediction_rgb[:,:,0] = overlay_crack

        # overlayed_prediction = image_file
        # overlayed_prediction[np.where(overlay_crack == 255)[0][0], np.where(overlay_crack==255)[0][1], 0] = 255
        # overlayed_prediction[np.where(overlay_crack == 255)[0][0], np.where(overlay_crack == 255)[0][1],1:] = 0

        overlayed_prediction = cv.addWeighted(image_file, 0.5, prediction_rgb, 1.0, 0)
        skimage.io.imsave(result_path+overlay_name_save, overlayed_prediction)
        #Updating prediction dictionary
        crack_prediction_bin[key] = overlay_crack
    
    if out_return:
        return crack_prediction_bin

#Testing
data_folder = "your_folder/"
images_path = '../data/' + data_folder 
crack_segmentation(images_path, out_return=False)
