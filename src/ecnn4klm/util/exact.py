import torch
        

def klmsq2d(spin, t=1.0, J=1.0, filling=0.5):
    """
    
    input:
        spin (torch.tensor, shape: (data_num, Lx, Ly, 3))
        t (float): nearest neighbor hopping
        J (float): Hund coupling
        filling (float): filling
    output:
        E (torch.tensor, shape: (data_num))
        h (torch.tensor, shape: (data_num, Lx, Ly, 3))
        H (torch.tensor, shape: (data_num, 2LxLy, 2LxLy))
    """
    data_num, Lx, Ly, _ = spin.shape
    dtype = spin.dtype
    device = spin.device
    if dtype == torch.float32:
        ctype = torch.complex64
    elif dtype == torch.float64:
        ctype = torch.complex128

    sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]],dtype=ctype, device=device)
    sigma_y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]],dtype=ctype, device=device) 
    sigma_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]],dtype=ctype, device=device) 
    
    H = torch.zeros([data_num,2*Lx*Ly,2*Lx*Ly],dtype=ctype, device=device)
    h = torch.zeros([data_num,Lx,Ly,3],dtype=dtype, device=device)
    num_el = int(round(2*Lx*Ly*filling))
    
    
    for ix in range(Lx):
        for iy in range(Ly):
            H[:,2*(ix*Ly+iy),2*((ix+1)%Lx*Ly+iy)] += -t # +x up
            H[:,2*(ix*Ly+iy)+1,2*((ix+1)%Lx*Ly+iy)+1] += -t # +x down
            H[:,2*((ix+1)%Lx*Ly+iy),2*(ix*Ly+iy)] += -t # +x up (cc.)
            H[:,2*((ix+1)%Lx*Ly+iy)+1,2*(ix*Ly+iy)+1] += -t # +x down (cc.)
            H[:,2*(ix*Ly+iy),2*(ix*Ly+(iy+1)%Ly)] += -t # +y up
            H[:,2*(ix*Ly+iy)+1,2*(ix*Ly+(iy+1)%Ly)+1] += -t # +y down
            H[:,2*(ix*Ly+(iy+1)%Ly),2*(ix*Ly+iy)] += -t # +y up (cc.)
            H[:,2*(ix*Ly+(iy+1)%Ly)+1,2*(ix*Ly+iy)+1] += -t # +y down (cc.)

            H[:,2*(ix*Ly+iy)+1,2*(ix*Ly+iy)] += -0.5*(J*spin[:,ix,iy,0] + 1j*J*spin[:,ix,iy,1])
            H[:,2*(ix*Ly+iy),2*(ix*Ly+iy)+1] += -0.5*(J*spin[:,ix,iy,0] - 1j*J*spin[:,ix,iy,1])
            H[:,2*(ix*Ly+iy),2*(ix*Ly+iy)] += -0.5*J*spin[:,ix,iy,2]
            H[:,2*(ix*Ly+iy)+1,2*(ix*Ly+iy)+1] += 0.5*J*spin[:,ix,iy,2]

    
    val, vec = torch.linalg.eigh(H)
    
    fermi = torch.zeros(2*Lx*Ly, dtype=ctype, device=device)
    fermi[0:num_el] = 1
    for ix in range(Lx):
        for iy in range(Ly):       
            h[:,ix,iy,0] += 0.5*torch.einsum("ikl,km,iml,l->i",vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:].conj(),sigma_x,vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:],fermi).real * J
            h[:,ix,iy,1] += 0.5*torch.einsum("ikl,km,iml,l->i",vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:].conj(),sigma_y,vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:],fermi).real * J
            h[:,ix,iy,2] += 0.5*torch.einsum("ikl,km,iml,l->i",vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:].conj(),sigma_z,vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:],fermi).real * J
                
    E = val[:,0:num_el].sum(dim=1).real
    return E, h, H
    
def klmtri2d_gc(spin, t1=1.0, t3=0.0, J=1.0, mu=0.0):
    """
    
    input:
        spin (torch.tensor, shape: (data_num, Lx, Ly, 3))
        t1 (float): hopping (nearest neighbor)
        t3 (float): hopping (third nearest neighbor)
        J (float): Hund coupling
        mu (float): Fermi level
    output:
        E (torch.tensor, shape: (data_num))
        h (torch.tensor, shape: (data_num, Lx, Ly, 3))
        H (torch.tensor, shape: (data_num, 2LxLy, 2LxLy))
    """
    data_num, Lx, Ly, _ = spin.shape
    dtype = spin.dtype
    device = spin.device
    if dtype == torch.float32:
        ctype = torch.complex64
    elif dtype == torch.float64:
        ctype = torch.complex128

    sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]],dtype=ctype, device=device)
    sigma_y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]],dtype=ctype, device=device) 
    sigma_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]],dtype=ctype, device=device) 
    
    H = torch.zeros([data_num,2*Lx*Ly,2*Lx*Ly],dtype=ctype, device=device)
    h = torch.zeros([data_num,Lx,Ly,3],dtype=dtype, device=device)
    
    
    
    for ix in range(Lx):
        for iy in range(Ly):
            H[:,2*(ix*Ly+iy),2*((ix+1)%Lx*Ly+iy)] += -t1 # +x up
            H[:,2*(ix*Ly+iy)+1,2*((ix+1)%Lx*Ly+iy)+1] += -t1 # +x down
            H[:,2*((ix+1)%Lx*Ly+iy),2*(ix*Ly+iy)] += -t1 # +x up (cc.)
            H[:,2*((ix+1)%Lx*Ly+iy)+1,2*(ix*Ly+iy)+1] += -t1 # +x down (cc.)
            H[:,2*(ix*Ly+iy),2*(ix*Ly+(iy+1)%Ly)] += -t1 # +y up
            H[:,2*(ix*Ly+iy)+1,2*(ix*Ly+(iy+1)%Ly)+1] += -t1 # +y down
            H[:,2*(ix*Ly+(iy+1)%Ly),2*(ix*Ly+iy)] += -t1 # +y up (cc.)
            H[:,2*(ix*Ly+(iy+1)%Ly)+1,2*(ix*Ly+iy)+1] += -t1 # +y down (cc.)
            H[:,2*(ix*Ly+iy),2*((ix+1)%Lx*Ly+(iy+1)%Ly)] += -t1 # +x+y up
            H[:,2*(ix*Ly+iy)+1,2*((ix+1)%Lx*Ly+(iy+1)%Ly)+1] += -t1 # +x+y down
            H[:,2*((ix+1)%Lx*Ly+(iy+1)%Ly),2*(ix*Ly+iy)] += -t1 # +x+y up (cc.)
            H[:,2*((ix+1)%Lx*Ly+(iy+1)%Ly)+1,2*(ix*Ly+iy)+1] += -t1 # +x+y down (cc.)

            H[:,2*(ix*Ly+iy),2*((ix+2)%Lx*Ly+iy)] += -t3 # +2x up
            H[:,2*(ix*Ly+iy)+1,2*((ix+2)%Lx*Ly+iy)+1] += -t3 # +2x down
            H[:,2*((ix+2)%Lx*Ly+iy),2*(ix*Ly+iy)] += -t3 # +2x up (cc.)
            H[:,2*((ix+2)%Lx*Ly+iy)+1,2*(ix*Ly+iy)+1] += -t3 # +2x down (cc.)
            H[:,2*(ix*Ly+iy),2*(ix*Ly+(iy+2)%Ly)] += -t3 # +2y up
            H[:,2*(ix*Ly+iy)+1,2*(ix*Ly+(iy+2)%Ly)+1] += -t3 # +2y down
            H[:,2*(ix*Ly+(iy+2)%Ly),2*(ix*Ly+iy)] += -t3 # +2y up (cc.)
            H[:,2*(ix*Ly+(iy+2)%Ly)+1,2*(ix*Ly+iy)+1] += -t3 # +2y down (cc.)
            H[:,2*(ix*Ly+iy),2*((ix+2)%Lx*Ly+(iy+2)%Ly)] += -t3 # +2x+2y up
            H[:,2*(ix*Ly+iy)+1,2*((ix+2)%Lx*Ly+(iy+2)%Ly)+1] += -t3 # +2x+2y down
            H[:,2*((ix+2)%Lx*Ly+(iy+2)%Ly),2*(ix*Ly+iy)] += -t3 # +2x+2y up (cc.)
            H[:,2*((ix+2)%Lx*Ly+(iy+2)%Ly)+1,2*(ix*Ly+iy)+1] += -t3 # +2x+2y down (cc.)

            H[:,2*(ix*Ly+iy)+1,2*(ix*Ly+iy)] += -0.5*(J*spin[:,ix,iy,0] + 1j*J*spin[:,ix,iy,1])
            H[:,2*(ix*Ly+iy),2*(ix*Ly+iy)+1] += -0.5*(J*spin[:,ix,iy,0] - 1j*J*spin[:,ix,iy,1])
            H[:,2*(ix*Ly+iy),2*(ix*Ly+iy)] += -0.5*J*spin[:,ix,iy,2] - mu
            H[:,2*(ix*Ly+iy)+1,2*(ix*Ly+iy)+1] += 0.5*J*spin[:,ix,iy,2] - mu
   
    val, vec = torch.linalg.eigh(H)
    fermi = (val.real < 0).to(ctype)
    # print(H)
    # print(val)
    # print(vec)
    for ix in range(Lx):
        for iy in range(Ly):
        
            h[:,ix,iy,0] += 0.5*torch.einsum("ikl,km,iml,il->i",vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:].conj(),sigma_x,vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:],fermi).real * J
            h[:,ix,iy,1] += 0.5*torch.einsum("ikl,km,iml,il->i",vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:].conj(),sigma_y,vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:],fermi).real * J
            h[:,ix,iy,2] += 0.5*torch.einsum("ikl,km,iml,il->i",vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:].conj(),sigma_z,vec[:,2*(ix*Ly+iy):2*(ix*Ly+iy)+2,:],fermi).real * J
                
   
    E = torch.einsum("il,il->i",val,fermi.real)
    return E, h, H

