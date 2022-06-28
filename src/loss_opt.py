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


import torch
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


##LOSS FUNCTION

#calculate the dice metric
def dice_loss(pred, target, smooth = 1e-5):

    target = target>=20 #as distance map needs to be binarized to work with dice loss
    pred = pred>=20 #as distance map needs to be binarized to work with dice loss

    intersection = (pred * target).sum(dim=(2,3))
    union= pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) 
    
    dice= 2.0 * (intersection + smooth) / (union+ smooth)    
    loss = 1.0 - dice
    
    return loss.sum(), dice.sum()


#dice loss function
def loss_func(pred, target):
       
    pred= torch.sigmoid(pred)
    dlv, _ = dice_loss(pred, target)
     
    loss = dlv
    #print(loss)
    
    return loss


##Other metrics
def dice_score(pred, target, slack=5, smooth = 1e-5):

    distances_gt = target.cpu()
    distances_pred = []
    
    for p in pred:
        
        p = p.cpu().detach().numpy()
        p = p<2
        p = skeletonize(p)
        distances_pred_array = ndimage.distance_transform_edt((np.logical_not(p)))
        distances_pred.append(distances_pred_array)
    
    distances_pred = torch.FloatTensor(distances_pred)
    
    ppred = distances_pred <= slack
    ttarget = distances_gt < slack
    
    
    intersection = (ppred * ttarget).sum(dim=(2,3))
    union= ppred.sum(dim=(2,3)) + ttarget.sum(dim=(2,3)) 
    
    dice= 2.0 * (intersection + smooth) / (union+ smooth)    
    
    return dice.sum().item()

def correctness(TP, FP, eps=1e-12):
    return TP/(TP + FP + eps) # precision

def completeness(TP, FN, eps=1e-12):
    return TP/(TP + FN + eps) # recall
    
def quality(TP, FP, FN, eps=1e-12):
    return TP/(TP + FP + FN + eps)
    
def f1(correctness, completeness, eps=1e-12):
    return 2.0/(1.0/(correctness+eps) + 1.0/(completeness+eps))

def relaxed_confusion_matrix(pred_s, gt_s, slack=5):
    
    distances_gt = gt_s.cpu()
    
    distances_pred = []
    pred_s_sk = []
    for p in pred_s:
        
        p = p.cpu().detach().numpy()
        p = p<2
        p = skeletonize(p)
        pred_s_sk.append(p)
        distances_pred_array = ndimage.distance_transform_edt((np.logical_not(p)))
        distances_pred.append(distances_pred_array)
    
    distances_pred = torch.FloatTensor(distances_pred)
    pred_s_sk = torch.FloatTensor(pred_s_sk)>0
    
    true_pos_area = distances_gt<=slack
    false_pos_area = distances_gt>slack   
    false_neg_area = distances_pred>slack
        
    true_positives = (true_pos_area*pred_s_sk).sum(dim=(2,3))
    false_positives = (false_pos_area*pred_s_sk).sum(dim=(2,3))
    
    false_negatives = (false_neg_area*((gt_s==0).cpu())).sum(dim=(2,3))
    
    return true_positives.sum().item(), false_negatives.sum().item(), false_positives.sum().item()

def correctness_completeness_quality(pred_s, gt_s, slack=5, eps=1e-12):
    
    TP, FN, FP = relaxed_confusion_matrix(pred_s, gt_s, slack)

    return correctness(TP, FP, eps), completeness(TP, FN, eps), quality(TP, FP, FN, eps)



#Define the metrics_batch helper function:
def metrics_batch(pred, target):
        
    metric_dice = dice_score(pred, target)
    correct, complet, qual = correctness_completeness_quality(pred, target)
    metric_f1 = f1(correct, complet)
    
    #Multiplied times len(pred) as during the training it is divided by the total of data
    return metric_dice, len(pred)*correct, len(pred)*complet, len(pred)*qual, len(pred)*metric_f1

##OPTIMIZER#
#for Learning scheduler get_lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
