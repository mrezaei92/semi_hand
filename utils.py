import torch
import os
import numpy as np

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
    

    Xs=x.flatten()[None,None,...]#.to(device)
    Ys=y.flatten()[None,None,...]#to(device)
    Xs.requires_grad=False
    Ys.requires_grad=False


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