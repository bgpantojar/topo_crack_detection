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
import torch
from loss_opt import *
import copy
import json
from tqdm import tqdm

#Helper functions to train model
#Loss by batch function:
def loss_batch(loss_f1,loss_f2,loss_f,malis_params,model_name, output, target, opt=None):   
    if model_name=='mse+topo':   
        loss = loss_f1(output.float(), target.float()) + loss_f2(output,target,malis_params[0],malis_params[1])
    elif model_name=='mse':
        loss = loss_f1(output.float(), target.float())
    elif model_name=='topo':
        loss = loss_f2(output,target,malis_params[0],malis_params[1])
    elif model_name=='dice':
        loss = loss_f(output.float(), target.float())
    elif model_name=='dice+topo':
        loss = loss_f(output.float(), target.float())+loss_f2(output,target,malis_params[0],malis_params[1])

    metric_b=torch.Tensor(metrics_batch(output, target))
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

#Define loss by epoch function:
def loss_epoch(model,loss_f1,loss_f2,loss_f,malis_params,model_name,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data=len(dataset_dl.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for xb, yb in tqdm(dataset_dl):
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)
        loss_b, metric_b=loss_batch(loss_f1,loss_f2,loss_f,malis_params,model_name, output, yb, opt)
        running_loss += loss_b
        
        if metric_b is not None:
            running_metric+=metric_b

        if sanity_check is True:
            break
    
    loss=running_loss/float(len_data)
    
    metric=running_metric/float(len_data)
    
    return loss, metric

#Train and validation function
def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func1=params["loss_func1"]
    loss_func2=params["loss_func2"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    path2weights=params["path2weights"]
    path2losshist=params["path2losshist"]
    path2metrichist=params["path2metrichist"]
    malis_params=params["malis_params"]
    model_name=params["model_name"]
    
    loss_history={
        "train": [],
        "val": []}
    
    dice_history={
        "train": [],
        "val": []}

    correctness_history={
        "train": [],
        "val": []}

    completeness_history={
        "train": [],
        "val": []}    

    quality_history={
        "train": [],
        "val": []}    

    f1_history={
        "train": [],
        "val": []}

    metric_history={}        
    
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')    
    
    best_dice=float(0)
    best_corr=float(0)
    best_comp=float(0)
    best_qual=float(0)
    best_f1=float(0)
    
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))   

        model.train()
        train_loss, train_metric = loss_epoch(model,loss_func1,loss_func2,loss_func,malis_params,model_name,train_dl,sanity_check,opt)

        loss_history["train"].append(train_loss)
        dice_history["train"].append(train_metric[0].item())
        correctness_history["train"].append(train_metric[1].item())
        completeness_history["train"].append(train_metric[2].item())
        quality_history["train"].append(train_metric[3].item())
        f1_history["train"].append(train_metric[4].item())
        
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model,loss_func1,loss_func2,loss_func,malis_params,model_name,val_dl,sanity_check)
       
        loss_history["val"].append(val_loss)
        dice_history["val"].append(val_metric[0].item())
        correctness_history["val"].append(val_metric[1].item())
        completeness_history["val"].append(val_metric[2].item())
        quality_history["val"].append(val_metric[3].item())
        f1_history["val"].append(val_metric[4].item())
        
        if val_loss < best_loss:
            best_loss = copy.deepcopy(val_loss)
            best_model_wts = copy.deepcopy(model.state_dict())            
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
            
        if val_metric[0].item() > best_dice:
            best_dice = copy.deepcopy(val_metric[0].item())
            torch.save(model.state_dict(), path2weights[:-3]+"_best_dice.pt")
            print("Copied best_dice model weights!")
        
        if val_metric[1].item() > best_corr:
            best_corr = copy.deepcopy(val_metric[1].item())
            torch.save(model.state_dict(), path2weights[:-3]+"_best_corr.pt")
            print("Copied best_corr model weights!")
            
        if val_metric[2].item() > best_comp:
            best_comp = copy.deepcopy(val_metric[2].item())
            torch.save(model.state_dict(), path2weights[:-3]+"_best_comp.pt")
            print("Copied best_comp model weights!")
            
        if val_metric[3].item() > best_qual:
            best_qual = copy.deepcopy(val_metric[3].item())
            torch.save(model.state_dict(), path2weights[:-3]+"_best_qual.pt")
            print("Copied best_qual model weights!")
            
        if val_metric[4].item() > best_f1:
            best_f1 = copy.deepcopy(val_metric[4].item())
            torch.save(model.state_dict(), path2weights[:-3]+"_best_f1.pt")
            print("Copied best_f1 model weights!")            
            
        print("train loss: %.6f, dice: %.2f, corr: %.2f, comp: %.2f, qual: %.2f, f1: %.2f" \
              %(train_loss, 100*train_metric[0].item(), 100*train_metric[1].item() \
                                              , 100*train_metric[2].item(), 100*train_metric[3].item(), \
                                                  100*train_metric[4].item()))
        print("val loss: %.6f, dice: %.2f, corr: %.2f, comp: %.2f, qual: %.2f, f1: %.2f" \
              %(val_loss, 100*val_metric[0].item(), 100*val_metric[1].item()\
                                            , 100*val_metric[2].item(), 100*val_metric[3].item()\
                                                , 100*val_metric[4].item()))
        print("-"*10) 
        
       
        with open(path2losshist,"w") as fp:
            json.dump(loss_history, fp)

        metric_history['dice'] = dice_history
        metric_history['corr'] = correctness_history
        metric_history['comp'] = completeness_history
        metric_history['qual'] = quality_history
        metric_history['f1'] = f1_history

        with open(path2metrichist,"w") as fp:
            json.dump(metric_history, fp)

    torch.save(model.state_dict(), path2weights[:-3]+"_last.pt")
    print("Copied last model weights!") 


    model.load_state_dict(best_model_wts)
    
    return model, loss_history, metric_history