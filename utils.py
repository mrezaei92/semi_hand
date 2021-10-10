import torch
import os
import numpy as np


def loss_masked(preds,target,mask,lossFunc):

    #preds, targets: both tensor of shape (B,num_joint,k)
    # mask a tensor of shape (B,num_joint,1)

    joint_dim=preds.shape[2]
    distance = lossFunc(preds,target) # B,num_joints,k
    distance=distance*mask

    num_noneZero = torch.sum(mask)*joint_dim
    if num_noneZero == 0:
        num_noneZero=1
    
    return torch.sum(distance)/num_noneZero


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])



def compute_entropy(x):
    # x is a tensor of size (b,k,dim1,dim2)
    # each map of size (dim1,dim2) should be a probability map (their value sum up to one)
    
    x=x.view(x.shape[0],x.shape[1],-1)
    logs=torch.log(x)
    return torch.sum(-1*x*logs,dim=-1)



def compute_MeanSTD(x):
    # x is a tensor of size (b,k,dim,dim), whould be a notmalized heatmap
    # mean will be a tensor of size (b,k,2), mean in X and Y dimension repectively
    # std will # (b,k,1)
    
    h0=x
    num_batch=x.shape[0]
    k=h0.shape[1]

    dim=x.shape[2]
    y,x=torch.meshgrid(torch.arange(0,dim),torch.arange(0,dim))
    

    Xs=x.flatten()[None,None,...].to(h0.device)#.to(device)
    Ys=y.flatten()[None,None,...].to(h0.device)#to(device)

    X0=torch.mul(h0.view(num_batch,k,-1),Xs)
    X_mean=torch.sum(X0,dim=-1)
    Y0=torch.mul(h0.view(num_batch,k,-1),Ys)
    Y_mean=torch.sum(Y0,dim=-1)

    X0=torch.unsqueeze(X_mean,dim=-1)
    Y0=torch.unsqueeze(Y_mean,dim=-1)
    mean=torch.cat((X0,Y0),dim=-1)

    X=Xs.repeat(1,k,1)
    X0=torch.mul(h0.view(num_batch,k,-1),(X-X_mean.unsqueeze(2))**2)
    X_std=torch.sum(X0,dim=-1)

    Y=Ys.repeat(1,k,1)
    Y0=torch.mul(h0.view(num_batch,k,-1),(Y-Y_mean.unsqueeze(2))**2)
    Y_std=torch.sum(Y0,dim=-1)
    
    X0=torch.unsqueeze(X_std,dim=-1)
    Y0=torch.unsqueeze(Y_std,dim=-1)
    variance=torch.cat((X0,Y0),dim=-1)
    std=torch.sum(variance,dim=-1)
    std=torch.sqrt(std)

    return (mean,std)


def print_tensor(x):
    # x should be a one-dimensional tensor
    s=''
    for i in range(x.shape[0]):
        s=s+f"{i}|{x[i]:.3f} "
        
    return s



def Signal_Annealing(progress,start,end,typee="cosine"):
    # progress is a number starting from 0 to reach 1 in the final step
    # start and end are starting and the end value respectively
    if typee == "cosine":
        return start + 0.5*(1 - np.cos(np.pi*progress) ) * (end-start)
    
    elif typee == "linear":
        return start + progress * (end-start)
    
    elif typee == "exp":
        return start + np.exp((progress-1)*5) * (end-start)
    
    elif typee == "log":
        return start + (1-np.exp((-progress)*5)) * (end-start)

    else:
        raise NotImplementedError
        
## Use case
# total=500
# start=0.1
# end=0.8
# x=[Signal_Annealing(i/total,start,end,"cosine") for i in range(total)]
# y=[Signal_Annealing(i/total,start,end,"linear") for i in range(total)]
# z=[Signal_Annealing(i/total,start,end,"exp") for i in range(total)]
# plt.plot(x)
# plt.plot(y)
# plt.plot(z)