import torch
import numpy as np
from math import sqrt
import random 
import warnings


def random_spin(data_num,lattice_size,dtype=torch.float32,device=torch.device("cpu")):
    """generate random spin configuration
    Args:
        data_num (int): number of data
        lattice_size (tupple): lattice size
        dtype (torch.dtype): data type
        device (torch.device): device type
    Returns:
        spin_data (torch.tensor, shape: (data_num, *lattice_size, 3): 
    """
    
    theta_list = torch.rand(data_num,*lattice_size) * torch.pi
    phi_list = torch.rand(data_num,*lattice_size) * 2*torch.pi
    spin_data =  torch.stack([torch.sin(theta_list)*torch.cos(phi_list), torch.sin(theta_list)*torch.sin(phi_list), torch.cos(theta_list)],dim=len(lattice_size)+1).to(dtype=dtype,device=device)
    return spin_data

def rotate_spin_tri2d(spin,rotation_fold="all",fold_list=None):
    warnings.warn("rotate_spin_tri2d is deprecated. Use RotateSpinTri2d Class", UserWarning)
    _, L1, L2, _ = spin.shape
    if rotation_fold=="all":
        spin_init = spin.detach().clone()
        spin_rotated = torch.zeros_like(spin_init)
        spin_list = [spin_init]
        for iter in range(5):
            for i1 in range(L1):
                for i2 in range(L2):
                    spin_rotated[:,(i1-i2)%L1,i1,:] = spin_init[:,i1,i2,:]
            
            
            spin_init = spin_rotated.clone()            
            spin_list.append(spin_rotated.clone())
            
        return torch.cat(spin_list, dim=0)
    
    elif rotation_fold=="single":
        spin_init = spin.detach().clone()
        spin_rotated = torch.zeros_like(spin_init)
        for i1 in range(L1):
            for i2 in range(L2):
                spin_rotated[:,(i1-i2)%L1,i1,:] = spin_init[:,i1,i2,:]
    
        return spin_rotated
    
    elif rotation_fold=="fold_list":
        spin_init = spin.detach().clone()
        
        spin_rotated = spin_init.clone()
        for fold in range(6):
            indexes = [i for i, e in enumerate(fold_list) if e == fold]
            for _ in range(fold):
                for i1 in range(L1):
                    for i2 in range(L2):            
                    
                        spin_rotated[indexes,(i1-i2)%L1,i1,:] = spin_init[indexes,i1,i2,:]
                spin_init = spin_rotated.clone()
        return spin_init
    else:
        print("rotation error")

class RotateSpinTri2d:
    def __init__(self,size:tuple):
        self.L1, self.L2 = size
        original_list = list(range(self.L1*self.L2))
        self.replace_list_list = [original_list.copy()]
        for _ in range(5):
            replace_list = []
            for i in original_list:
                i1 = i // self.L2
                i2 = i % self.L2
                replace_list.append(i2*self.L2+(i2-i1)%self.L2)
            original_list = replace_list.copy()
            self.replace_list_list.append(replace_list.copy())
    def rotate(self,spin,fold):
        return spin.reshape(-1,self.L1*self.L2,3)[:,self.replace_list_list[fold],:].reshape(-1,self.L1,self.L2,3)
    def rotate_list(self,spin,fold_list):
        assert spin.shape[0] == len(fold_list)
        for i in range(len(fold_list)):
            fold = fold_list[i]
            spin[i] = spin[i].reshape(self.L1*self.L2,3)[self.replace_list_list[fold],:].reshape(self.L1,self.L2,3)
        return spin

def variational_tri2d(Q,theta,psi,az,Mz,size,dtype=torch.float32,device=torch.device("cuda")):
    L1, L2 = size
    Q1 = [Q, 0]
    Q2 = [-0.5*Q, sqrt(3)/2]
    Q3 = [-0.5*Q, -sqrt(3)/2]
    spin = torch.zeros(L1,L2,3,dtype=dtype,device=device)
    a1,a2 = torch.meshgrid(torch.tensor(range(L1),dtype=dtype,device=device),torch.tensor(range(L2),dtype=dtype,device=device),indexing="xy")
    x = a1 - 0.5 * a2 
    y = sqrt(3) / 2.0 * a2
    spin[:,:,0] += torch.sin(Q1[0]*x + Q1[1]*y + theta[0] + psi)
    spin[:,:,0] += -0.5 * torch.sin(Q2[0]*x + Q2[1]*y + theta[1] + psi)
    spin[:,:,0] += -0.5 * torch.sin(Q3[0]*x + Q3[1]*y + theta[2] + psi)
    spin[:,:,1] += sqrt(3) / 2.0 * torch.sin(Q2[0]*x + Q2[1]*y + theta[1] + psi)
    spin[:,:,1] += -sqrt(3) / 2.0 * torch.sin(Q3[0]*x + Q3[1]*y + theta[2] + psi)
    spin[:,:,2] += Mz
    spin[:,:,2] -= az * (torch.cos(Q1[0]*x + Q1[1]*y + theta[0]))
    spin[:,:,2] -= az * (torch.cos(Q2[0]*x + Q2[1]*y + theta[1]))
    spin[:,:,2] -= az * (torch.cos(Q3[0]*x + Q3[1]*y + theta[2]))
    spin_norm = torch.norm(spin,dim=2,p=2)
    spin[:,:,0] /= spin_norm
    spin[:,:,1] /= spin_norm
    spin[:,:,2] /= spin_norm
    return spin
