import e3nn
import torch
from e3nn import io, o3
import e3nn.nn
import numpy as np
import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Irrep
import torch.nn.functional as F

class SpinConvSq2d(nn.Module):
    def __init__(self, ir_in, l_f, ir_out, kernel_size, p_val=1, p_arg=1):
        '''
        Args:
            ir_in (int): irreducible representations of input feature
            l_f (int): lotational order of filter feature 
            ir_out (int): irreducible representations of output feature
            kernel_size (int): kernel size (2*kernel_size+1)*(2*kernel_size+1)
        '''
        super().__init__()

        self.kernel_size = kernel_size
        self.l_f = l_f
        ir_in = o3.Irreps(ir_in)
        ir_in = ir_in * (2*kernel_size+1)**2 
        ir_out = o3.Irreps(ir_out)
        ir_filter = io.SphericalTensor(self.l_f, p_val=p_val, p_arg=p_arg) 

        self.tp = o3.TensorProduct(ir_in,ir_filter,ir_out,
            [
                (a,b,c,"uvw",True)
                for a in range(len(ir_in))
                for b in range(len(ir_filter))
                for c in range(len(ir_out))
                if (abs(ir_in[a][1].l-ir_filter[b][1].l)<=ir_out[c][1].l<=ir_in[a][1].l+ir_filter[b][1].l) and (ir_in[a][1].p*ir_filter[b][1].p==ir_out[c][1].p)
            ]
        )
        

    def forward(self, feat, spin):
        """
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in))
            spin (torch.tensor, shape: (data_num, Lx, Ly, 3))
        Returns:
            feat (torch.tensor, shape: (data_num, channel, Lx, Ly, feature_dims_out))
        """
        
        spin = o3.spherical_harmonics(list(range(self.l_f+1)),spin,normalize=False)
        feat = self.spin2colsq2d(feat)
        feat = self.tp(feat, spin)

        return feat


    def spin2colsq2d(self,feat):
        """
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in))
            kernel_size (positive int)
        Returns:
            col (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in*(2*kernel_size+1)))
        """
        N, Lx, Ly, f_num = feat.shape
        feat = feat.permute(0,3,1,2)
        feat = F.pad(feat, (self.kernel_size,self.kernel_size,self.kernel_size,self.kernel_size),"circular")
        feat = feat.permute(0,2,3,1)
        col = torch.zeros([N,Lx,Ly,2*self.kernel_size+1,2*self.kernel_size+1,f_num],device=feat.device)
        for ix in range(2*self.kernel_size+1):
            for iy in range(2*self.kernel_size+1):
                col[:,:,:,ix,iy,:] = feat[:,ix:ix+Lx,iy:iy+Ly,:]
        return col.reshape(N,Lx,Ly,-1)

class SpinConvSq2dSparse(nn.Module):
    def __init__(self, ir_in, l_f, ir_out, kernel_size, p_val=1, p_arg=1):
        '''
        Args:
            ir_in (int): irreducible representations of input feature
            l_f (int): lotational order of filter feature 
            ir_out (int): irreducible representations of output feature
            kernel_size (int): kernel size (4*kernel_size+1)
        '''
        super().__init__()

        self.kernel_size = kernel_size
        self.l_f = l_f
        ir_in = o3.Irreps(ir_in)
        ir_in = ir_in * (4*kernel_size+1) 
        ir_out = o3.Irreps(ir_out)
        ir_filter = io.SphericalTensor(self.l_f, p_val=p_val, p_arg=p_arg) 

        self.tp = o3.TensorProduct(ir_in,ir_filter,ir_out,
            [
                (a,b,c,"uvw",True)
                for a in range(len(ir_in))
                for b in range(len(ir_filter))
                for c in range(len(ir_out))
                if (abs(ir_in[a][1].l-ir_filter[b][1].l)<=ir_out[c][1].l<=ir_in[a][1].l+ir_filter[b][1].l) and (ir_in[a][1].p*ir_filter[b][1].p==ir_out[c][1].p)
            ]
        )
        

    def forward(self, feat, spin):
        """
        Args:
            feat (torch.tensor, shape: (data_num, 1, Lx, Ly, feature_dims_in))
            spin (torch.tensor, shape: (data_num, 1, Lx, Ly, 3))
        Returns:
            feat (torch.tensor, shape: (data_num, channel, Lx, Ly, feature_dims_out))
        """
        
        spin = o3.spherical_harmonics(list(range(self.l_f+1)),spin,normalize=False)
        feat = self.spin2colsq2d_sparse(feat)
        feat = self.tp(feat, spin)

        return feat


    def spin2colsq2d_sparse(self,feat):
        """
        
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in))
            kernel_size (positive int)
        Returns:
            col (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in*(4*kernel_size+1)))
        """
        N, Lx, Ly, f_num = feat.shape
        feat = feat.permute(0,3,1,2)
        feat = F.pad(feat, (self.kernel_size,self.kernel_size,self.kernel_size,self.kernel_size),"circular")
        feat = feat.permute(0,2,3,1)
        col = torch.zeros([N,Lx,Ly,4*self.kernel_size+1,f_num],device=feat.device)
        col[:,:,:,0,:] = feat[:,self.kernel_size:self.kernel_size+Lx,self.kernel_size:self.kernel_size+Ly,:]
        for i in range(self.kernel_size):
            col[:,:,:,4*i+1,:] = feat[:,self.kernel_size-i:self.kernel_size-i+Lx,self.kernel_size:self.kernel_size+Ly,:]
            col[:,:,:,4*i+2,:] = feat[:,self.kernel_size+i:self.kernel_size+i+Lx,self.kernel_size:self.kernel_size+Ly,:]
            col[:,:,:,4*i+3,:] = feat[:,self.kernel_size:self.kernel_size+Lx,self.kernel_size-i:self.kernel_size-i+Ly,:]
            col[:,:,:,4*i+4,:] = feat[:,self.kernel_size:self.kernel_size+Lx,self.kernel_size+i:self.kernel_size+i+Ly,:]
             
        return col.reshape(N,Lx,Ly,-1)

class SpinConvTri2d(nn.Module):
    def __init__(self, ir_in, l_f, ir_out, kernel_size, p_val=1, p_arg=1):
        '''
        Args:
            ir_in (Irreps): irreducible representations of input feature
            l_f (int): lotational order of filter feature 
            ir_out (Irreps): irreducible representations of output feature
            kernel_size (int): kernel size (3*kernel_size**2+3*kernel_size+1) 
        '''
        super().__init__()

        self.kernel_size = kernel_size
        self.l_f = l_f
        ir_in = o3.Irreps(ir_in)
        ir_in = ir_in * (3*kernel_size**2+3*kernel_size+1) 
        ir_out = o3.Irreps(ir_out)
        ir_filter = io.SphericalTensor(self.l_f, p_val=p_val, p_arg=p_arg) 
        self.tp = o3.TensorProduct(ir_in,ir_filter,ir_out,
            [
                (a,b,c,"uvw",True)
                for a in range(len(ir_in))
                for b in range(len(ir_filter))
                for c in range(len(ir_out))
                if (abs(ir_in[a][1].l-ir_filter[b][1].l)<=ir_out[c][1].l<=ir_in[a][1].l+ir_filter[b][1].l) and (ir_in[a][1].p*ir_filter[b][1].p==ir_out[c][1].p)
            ]
        )
        

    def forward(self, feat, spin):
        """
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in))
            spin (torch.tensor, shape: (data_num, Lx, Ly, 3))
        Returns:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_out))
        """
        
        spin = o3.spherical_harmonics(list(range(self.l_f+1)),spin,normalize=False)
        feat = self.spin2coltri2d(feat)
        feat = self.tp(feat, spin)

        return feat


    def spin2coltri2d(self,feat):
        """
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in))
            kernel_size (positive int)
        Returns:
            col (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in*(3*kernel_size**2+3*kernel_size+1)))
        """
        N, Lx, Ly, f_num = feat.shape
        feat = feat.permute(0,3,1,2)
        feat = F.pad(feat, (self.kernel_size,self.kernel_size,self.kernel_size,self.kernel_size),"circular")
        feat = feat.permute(0,2,3,1)
        
        hexagon = [[1,1],[1,0],[0,-1],[-1,-1],[-1,0],[0,1]]
        col = torch.zeros([N,Lx,Ly,3*self.kernel_size**2+3*self.kernel_size+1,f_num],device=feat.device)
        
        for k in range(1,self.kernel_size+1): 
            for l in range(6*k): 
                i = 3*(k-1)**2 + 3*(k-1) + 1 + l
                edge_before = hexagon[l//k]
                edge_after = hexagon[(l//k+1)%6]
                ix = edge_before[0]*(k-l%k) + edge_after[0]*(l%k) + self.kernel_size
                iy = edge_before[1]*(k-l%k) + edge_after[1]*(l%k) + self.kernel_size
                col[:,:,:,i,:] = feat[:,ix:ix+Lx,iy:iy+Ly,:]
        #k=0
        col[:,:,:,0,:] = feat[:,self.kernel_size:self.kernel_size+Lx,self.kernel_size:self.kernel_size+Ly,:]
        
        return col.reshape(N,Lx,Ly,-1)

class SpinConvTri2dSparse(nn.Module):
    def __init__(self, ir_in, l_f, ir_out, kernel_size, p_val=1, p_arg=1):
        '''
        Args:
            ir_in (Irreps): irreducible representations of input feature
            l_f (int): lotational order of filter feature 
            ir_out (Irreps): irreducible representations of output feature
            kernel_size (int): kernel size (6*kernel_size+1) 
        '''
        super().__init__()

        self.kernel_size = kernel_size
        self.l_f = l_f
        ir_in = o3.Irreps(ir_in)
        ir_in = ir_in * (6*kernel_size+1) 

        ir_out = o3.Irreps(ir_out)
        ir_filter = io.SphericalTensor(self.l_f, p_val=p_val, p_arg=p_arg) 

        self.tp = o3.TensorProduct(ir_in,ir_filter,ir_out,
            [
                (a,b,c,"uvw",True)
                for a in range(len(ir_in))
                for b in range(len(ir_filter))
                for c in range(len(ir_out))
                if (abs(ir_in[a][1].l-ir_filter[b][1].l)<=ir_out[c][1].l<=ir_in[a][1].l+ir_filter[b][1].l) and (ir_in[a][1].p*ir_filter[b][1].p==ir_out[c][1].p)
            ]
        )
        

    def forward(self, feat, spin):
        """
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in))
            spin (torch.tensor, shape: (data_num, Lx, Ly, 3))
        Returns:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_out))
        """
        spin = o3.spherical_harmonics(list(range(self.l_f+1)),spin,normalize=False)
        feat = self.spin2coltri2d_sparse(feat)
        feat = self.tp(feat, spin)

        return feat
    
    def spin2coltri2d_sparse(self,feat):
        """
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in))
            kernel_size (positive int)
        Returns:
            col (torch.tensor, shape: (data_num, Lx, Ly, feature_dims_in*(6*kernel_size+1)))
        """
        N, Lx, Ly, f_num = feat.shape
        feat = feat.permute(0,3,1,2)
        feat = F.pad(feat, (self.kernel_size,self.kernel_size,self.kernel_size,self.kernel_size),"circular")
        feat = feat.permute(0,2,3,1)
        
        hexagon = [[1,1],[1,0],[0,-1],[-1,-1],[-1,0],[0,1]]
        col = torch.zeros([N,Lx,Ly,6*self.kernel_size+1,f_num],device=feat.device)
        
        for k in range(1,self.kernel_size+1): 
            for l in range(6): 
                i = 6*(k-1) + 1 + l
                ix = hexagon[l][0] * k + self.kernel_size
                iy = hexagon[l][1] * k + self.kernel_size
                col[:,:,:,i,:] = feat[:,ix:ix+Lx,iy:iy+Ly,:]
        #k=0
        col[:,:,:,0,:] = feat[:,self.kernel_size:self.kernel_size+Lx,self.kernel_size:self.kernel_size+Ly,:]
            
        return col.reshape(N,Lx,Ly,-1)

class SelfInt2d(nn.Module):
    """
    Args:
        l (int): lotational order 
        ir_in (Irreps): irreducible representations of input feature    
        ir_out (Irreps): irreducible representations of output feature       
        biases (bool): use biases in l=0 or not
    """
    def __init__(self, ir_in, ir_out, biases = False):
        super().__init__()
        ir_in = o3.Irreps(ir_in)
        ir_out = o3.Irreps(ir_out)
        self.selfinteract = e3nn.o3.Linear(ir_in, ir_out, biases = biases)

    def forward(self, feat):
        """
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dim))
        Returns:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dim))
        """
        feat = self.selfinteract(feat)
        
        return feat

class NonLinear2d(nn.Module):
    """
    Args:
        ir (Irreps): irreducible representations of input feature
        scalar_nonlinearity (function): activation function
        bias (bool): use biases in l=0 or not
    """
    def __init__(self, ir , scalar_nonlinearity = nn.SiLU(), bias = True):
        super().__init__()
        ir = o3.Irreps(ir)
        self.nact = e3nn.nn.NormActivation(ir, scalar_nonlinearity, bias=bias)

    def forward(self, feat):
        """
        Args:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dim))
        Returns:
            feat (torch.tensor, shape: (data_num, Lx, Ly, feature_dim))
        """
        
        feat = self.nact(feat)
        
        return feat


