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

# Import Modules
import os
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import scipy

random.seed(100)

#Define the open_dataset class:
class open_dataset(Dataset):
    def __init__(self, path2data_i, path2data_m=None, transform=None):      

        imgsList=[pp for pp in os.listdir(path2data_i)]
        imgsList.sort()
        if transform!="test":
            anntsList=[pp for pp in os.listdir(path2data_m)]
            anntsList.sort()

        self.path2imgs = [os.path.join(path2data_i, fn) for fn in imgsList] 
        
        if transform!="test":
            self.path2annts= [os.path.join(path2data_m, fn) for fn in anntsList]

        self.transform = transform
    
    def __len__(self):
        return len(self.path2imgs)
      
    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        img = Image.open(path2img).convert('RGB')
        if self.transform!="test":
            path2annt = self.path2annts[idx]
            mask = Image.open(path2annt)
                
        if self.transform=='train':
                if random.random()<.5:
                    img = TF.hflip(img)
                    mask = TF.hflip(mask)
                if random.random()<.5:
                    img = TF.vflip(img)
                    mask = TF.vflip(mask)
                if random.random()<.5:
                    img = TF.adjust_brightness(img,brightness_factor=.5)
                if random.random()<.5:
                    img = TF.adjust_contrast(img,contrast_factor=.4)
                if random.random()<.5:
                    img = TF.adjust_gamma(img,gamma=1.4)
                if random.random()<.5:
                    trans = T.Grayscale(num_output_channels=3)
                    img = trans(img)
                if random.random()<.0:
                    trans = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)
                    img = trans(img)
        
        if self.transform!='test':
            im_size = 256
            trans = T.Resize((im_size,im_size))
            img = trans(img)

        if self.transform!="test": mask = trans(mask)
        trans = T.ToTensor()
        img = trans(img)
        if self.transform!="test":
            mask = np.array(mask) #to array
            mask = 1-np.array(mask>0) #background zero. 1-mask
            mask=scipy.ndimage.distance_transform_edt(mask) #creating distance map
            mask[mask>20] = 20 #Cliping the distance map
            mask = trans(mask)
        
        #VGG16 mean and std 
        meanR, meanG, meanB = .485,.456,.406
        stdR, stdG, stdB = .229, .224, .225 
        norm_= T.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
        img = norm_(img)
        
        if self.transform!='test':
            return img, mask
        else:
            return img