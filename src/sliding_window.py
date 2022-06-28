# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:58:28 2020

This script contains the codes to segment cracks.
The codes are based on "TOPO-Loss for continuity-preserving crack detection using deep learning" 
by Pantoja-Rosero et., al.
https://doi.org/10.1016/j.conbuildmat.2022.128264

@author: rezaie
"""

## Load necessary packages and instances
import os
import numpy as np
import skimage.io

def zero_pad(image, desired_size): # imgae : RGB

    remainder_y = image.shape[0] % desired_size
    if remainder_y!=0:
        newImgSize_y = image.shape[0] + desired_size - remainder_y
    else:
        newImgSize_y = image.shape[0] 
    remainder_x = image.shape[1] % desired_size
    if remainder_x!=0:
        newImgSize_x = image.shape[1] + desired_size - remainder_x
    else:
        newImgSize_x = image.shape[1] 

    if image.ndim == 3:
        new_im = np.zeros((newImgSize_y, newImgSize_x, 3), dtype = "uint8")
        new_im[0:image.shape[0], 0:image.shape[1], :] = image[:,:,:]
        return new_im
    if image.ndim == 2:
        new_im = np.zeros((newImgSize_y, newImgSize_x), dtype = "uint8")
        new_im[0:image.shape[0], 0:image.shape[1]] = image[:,:]
        return new_im 

def sliding_window(image, stepSize, windowSize):
    if image.ndim == 3:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0], :])
    if image.ndim == 2:
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
                
# Load images
def main_sliding(data_set,type_data, save=True, desired_size = 256):
    image_path = '../data/'+data_set+'/'+type_data
    imageNames = []
    image_files = []
    image_patches = {}
    for imageFile in os.listdir(image_path):
        if imageFile[-4:] in ['.JPG','.tif','.png', '.jpg', '.PNG']: 
            imageNames.append(imageFile[:-4])
            image_files.append(skimage.io.imread(os.path.join(image_path, imageFile)))
            
    #  Chnage Grayscale sliced images to RGB and save them
    sliced_images_path = '../data/'+data_set+'/'+type_data+'_patches'

    check_dir = os.path.isdir(sliced_images_path)
    if not check_dir:
        os.makedirs(sliced_images_path)

    stepSize=desired_size
    for index_image in np.arange(len(image_files)):
        # First images are turned to rgb and also zero padded
        new_image = zero_pad(image_files[index_image], desired_size)
        index = 0
        image_patches[index_image] = {}

        for (x, y, window) in sliding_window(new_image, stepSize, windowSize=(desired_size, desired_size)):
            imageNames[index_image] = imageNames[index_image].replace("_mask","")
            if save:
                save_dir = os.path.join(sliced_images_path, imageNames[index_image]+ "_{:d}".format(index)+ "_{:d}".format(x) + "_{:d}".format(y)+ '.png')
                skimage.io.imsave(save_dir, window)
            else:
                image_patches[index_image][imageNames[index_image]+ "_{:d}".format(index)+ "_{:d}".format(x) + "_{:d}".format(y)+ '.png'] = window
            index += 1
    
    if not save:
        return image_patches