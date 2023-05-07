import sys
from ecnn4klm.util.spin_generator import random_spin, rotate_spin_tri2d, RotateSpinTri2d
from ecnn4klm.model import SpinConvTri2dSparse, NonLinear2d, SelfInt2d
import torch
from torch import nn
import e3nn
import pprint
import torch
from e3nn import io, o3

import numpy as np
import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Irrep


import os, sys, datetime, shutil, time, random, subprocess


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_i = 2
        l_f = 1
        k = 2
        l_hid = 2
        c = 8
        ir_hid = Irreps([(c,(l,1)) for l in range(l_hid+1)])
        

        self.conv1 = SpinConvTri2dSparse(io.SphericalTensor(self.l_i, p_val=1, p_arg=1), l_f, ir_hid, kernel_size=k)
        self.act1 = NonLinear2d(ir_hid, bias = False)
        self.conv2 = SpinConvTri2dSparse(ir_hid, l_f, ir_hid, kernel_size=k)
        self.act2 = NonLinear2d(ir_hid, bias = False)
        self.conv3 = SpinConvTri2dSparse(ir_hid, l_f, ir_hid, kernel_size=k)
        self.act3 = NonLinear2d(ir_hid, bias = False)
        self.conv4 = SpinConvTri2dSparse(ir_hid, l_f, ir_hid, kernel_size=k)
        self.act4 = NonLinear2d(ir_hid, bias = False)
        


        self.convout = SpinConvTri2dSparse(ir_hid, l_f, "8x0e", kernel_size=k)
        
        
        self.selfintout1 = SelfInt2d("8x0e", "4x0e", biases=True)
        self.actout1 = NonLinear2d("4x0e", bias = False)
        self.selfintout2 = SelfInt2d("4x0e", "2x0e", biases=True)
        self.actout2 = NonLinear2d("2x0e", bias = False)
        self.selfintout3 = SelfInt2d("2x0e", "1x0e", biases=True)



    def forward(self,spin):
        feat = e3nn.o3.spherical_harmonics(list(range(self.l_i+1)), spin, normalize=False)


        

        feat = self.conv1(feat,spin)
        feat_sk = self.act1(feat)
        feat = self.conv2(feat_sk,spin)
        feat_sk = self.act2(feat+feat_sk)
        feat = self.conv3(feat_sk,spin)
        feat_sk = self.act3(feat+feat_sk)    
        feat = self.conv4(feat_sk,spin)
        feat_sk = self.act4(feat+feat_sk)
        
        feat = self.convout(feat_sk,spin)
        
        

        
        feat = self.selfintout1(feat)
        feat = self.actout1(feat)
        feat = self.selfintout2(feat)
        feat = self.actout2(feat)
        feat = self.selfintout3(feat)
        return feat






if __name__ == "__main__":
    device = torch.device("cuda")
    params = {
        "early_stop": 500,
        "lr_min": 1e-5,
        "batch_size": 1,
        "seed": 123,
        "lamb": 1.0,
    }
    locals().update(params)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(seed)
    rotator = RotateSpinTri2d((Lx,Ly))
    #
    comment = f"""\
    
            
            
    """
    class Logger:
        def __init__(self, filename):
            self.console = sys.stdout
            self.file = open(filename, 'w')
        def write(self, message):
            self.console.write(message)
            self.file.write(message)
        def flush(self):
            self.console.flush()
            self.file.flush()
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    dir_for_output = os.path.join("./log-learn/", current_time + "/")
    os.makedirs(dir_for_output, exist_ok=True)
    logfile = "log.txt"
    sys.stdout = Logger(os.path.join(dir_for_output,logfile))
    shutil.copy(__file__, dir_for_output)
    print("#############################")
    res = subprocess.run("git rev-parse HEAD",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, shell=True)

    print(f"commit id: {res.stdout}")
    print(dir_for_output)
    print(comment)
    pprint.pprint(params)

    print("#############################")
    sys.stdout.flush()
    net = Net().to(device)
    optimizer = torch.optim.Adagrad(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True, threshold=0)

    print("#############################")
    print(net)
    print(optimizer)
    print(scheduler)
    print("#############################")
    sys.stdout.flush()
    #torch._C._jit_set_bailout_depth(0) # remove commentout if batch_size >= 2
    


    data_dict = torch.load("data.pt")
    spin_train = data_dict["spin_train"].to(device)
    E_train = data_dict["E_train"].to(device)
    B_train = data_dict["B_train"].to(device)
    spin_test = data_dict["spin_test"].to(device)
    E_test = data_dict["E_test"].to(device)
    B_test = data_dict["B_test"].to(device)
    


    ##########
    #learning
    ##########

    
    import csv
    import torch.utils.data as data
    
    
    train_set = data.TensorDataset(spin_train, E_train, B_train) 
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, generator=g)
    test_set = data.TensorDataset(spin_test, E_test, B_test) 
    test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    
    loss_func = torch.nn.MSELoss()
    
    
    train_losses = []
    train_losses_E = [] 
    train_losses_B = []
    test_losses = []
    test_losses_E = [] 
    test_losses_B = []
    with open(os.path.join(dir_for_output, "learn_data.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "test_loss", "train_loss_E", "test_loss_E", "train_loss_B", "test_loss_B"]) # with E
        time_start = time.perf_counter()
        best_epoch = 0
        for epoch in range(10000):
            net.train()
            train_loss = []
            train_loss_E = [] # with E
            train_loss_B = []
            for spin,E,B in train_loader: # with E
                fold = random.randrange(6)
                spin = rotator.rotate(spin,fold)
                B = rotator.rotate(B,fold)
                spin.requires_grad_(True)
                optimizer.zero_grad()

                E_pred = net(spin).sum(dim=(1,2,3))
                B_pred = -torch.autograd.grad(E_pred[0], spin, create_graph=True)[0]
                # B_pred = -torch.autograd.grad(E_pred, spin, torch.eye(batch_size,dtype=torch.float32).to(device), create_graph=True, is_grads_batched=True)[0].sum(dim=1)
                loss_E = loss_func(E,E_pred)/ (Lx*Ly)**2  
                loss_B = loss_func(B,B_pred) 
                loss = loss_B + loss_E * lamb
                train_loss.append(loss.item())
                train_loss_E.append(loss_E.item()) 
                train_loss_B.append(loss_B.item())
                
                loss.backward()
                optimizer.step()
            train_losses.append(sum(train_loss)/len(train_loss))
            train_losses_E.append(sum(train_loss_E)/len(train_loss_E)) # with E
            train_losses_B.append(sum(train_loss_B)/len(train_loss_B))

            test_loss = []
            test_loss_E = [] # with E
            test_loss_B = []
            net.eval()
            for spin,E,B in test_loader: # with E
            
                fold = random.randrange(6)
                spin = rotator.rotate(spin,fold)
                B = rotator.rotate(B,fold)
                spin.requires_grad_(True)
                E_pred = net(spin).sum(dim=(1,2,3))
                B_pred = -torch.autograd.grad(E_pred[0], spin, create_graph=True)[0]
                # B_pred = -torch.autograd.grad(E_pred, spin, torch.eye(batch_size,dtype=torch.float32).to(device), create_graph=True, is_grads_batched=True)[0].sum(dim=1)
                loss_E = loss_func(E,E_pred)/ (Lx*Ly)**2 # with E
                loss_B = loss_func(B,B_pred)
                loss = loss_B + loss_E * lamb
                test_loss.append(loss.item())
                test_loss_E.append(loss_E.item()) # with E
                test_loss_B.append(loss_B.item())
                
            test_losses.append(sum(test_loss)/len(test_loss))
            test_losses_E.append(sum(test_loss_E)/len(test_loss_E)) # with E
            test_losses_B.append(sum(test_loss_B)/len(test_loss_B))
            time_now = time.perf_counter()

            print(f"epoch {epoch}: lr:{optimizer.param_groups[0]['lr']:.3e}, train:{train_losses[epoch]}, test:{test_losses[epoch]}, train_E:{train_losses_E[epoch]}, test_E:{test_losses_E[epoch]}, train_B:{train_losses_B[epoch]}, test_B:{test_losses_B[epoch]}, elapsed: {time_now-time_start} sec.")
            writer.writerow([epoch, optimizer.param_groups[0]['lr'], train_losses[epoch], test_losses[epoch], train_losses_E[epoch], test_losses_E[epoch], train_losses_B[epoch], test_losses_B[epoch]]) # with E
            sys.stdout.flush()
            f.flush()
            torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'loss': test_losses[epoch],
                }, os.path.join(dir_for_output,"last.cpt"))
            if test_losses[epoch] == min(test_losses):
                torch.save({'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'loss': test_losses[epoch],
                    }, os.path.join(dir_for_output,"best.cpt"))
                print("saved")
                best_epoch = epoch
                best_train_loss = train_losses[epoch]
                best_test_loss = test_losses[epoch]
            if ((epoch - best_epoch) > early_stop) or (optimizer.param_groups[0]['lr'] < lr_min) :
                print(f"best model: epoch:{best_epoch}, train:{best_train_loss}, test:{best_test_loss}")
                break

            scheduler.step(test_losses[epoch])


        
        
