import torch
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def plot_spin_tri2d_color(spin,filename="fig_spin.png",dpi=300,figsize=(4,4),scale=None,width=None,fontsize=12,cmap="jet",window=None,close=True):
    """
    window: (x_min,y_min,width,height)
    """
    spin = spin.numpy()
    L1, L2, _ = spin.shape
    if scale is None:
        if window is None:
            scale = max(L1,L2)*1.2
        else:
            scale = max(window[2],window[3])*1.2
        
    if width is None:
        width = 1.0 / scale * 0.1
    
    
    plt.rcParams["font.size"] = fontsize
    a1,a2 = np.meshgrid(range(L1), range(L2), indexing = 'ij')
    x = (a1 - a2 * 0.5) % L1
    y = (a2 * sqrt(3.0)/2.0) % L2
    
    
    fig = plt.figure(figsize=figsize,tight_layout=True)
    ax = plt.gca()
    ax.set_aspect("equal")
    # PyPlot.pcolormesh(x_full, y_full, spin[1:Lx,1:Ly,3], clim=(-1.,1.), shading="gouraud", cmap="coolwarm")
    plt.quiver(x, y, spin[:,:,0], spin[:,:,1], spin[:,:,2],
        clim=(-1.,1.), scale=scale, pivot="middle", cmap=cmap, width=width)
    if not(window is None):
        ax.set_xlim(window[1],window[1]+window[3])
        ax.set_ylim(window[2],window[2]+window[4])
    
    if ax.get_position().height < ax.get_position().width + 0.1:
        cax = plt.colorbar(label=r"$S^z$",ax=ax, shrink=ax.get_position().height,ticks=[-1,0,1])
        
    else:
        cax = plt.colorbar(label=r"$S^z$",ax=ax, shrink=1.0,ticks=[-1,0,1])
       
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    
    plt.savefig(filename, dpi=dpi)
    if close:
        plt.close("all")
