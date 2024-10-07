import torch
import numpy as np

import os,sys
import time,datetime
from utils.ceof_generator import tri_occ
from utils.ceof_recon import Recon
import torch.nn.functional as F

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
    

def visualize_inter_res(fof_data,device,idx):
    e = Recon(device)

    with torch.no_grad():
        occupancy = e.vis_decode(fof_data[0])
        
        occupancy = torch.unsqueeze(occupancy,dim=0)
        occupancy = torch.unsqueeze(occupancy,dim=0)
        
        occu_down = F.interpolate(occupancy, size=[64,64,64],mode = 'trilinear')
        bool_ma = torch.ge(occu_down,0.5)
        index = (bool_ma[0][0] == True).nonzero(as_tuple=False)
        coords = index.cpu().numpy()
        x = coords[:,0]
        y = coords[:,1]
        z = coords[:,2]
        
        fig=plt.figure(dpi=120)
        ax=fig.add_subplot(111,projection='3d')
        ax.scatter(z,-x,-y,c='b',marker='.',s=2,linewidth=0,alpha=1,cmap='spectral')
        ax.axis('scaled')          
        ax.set_xlabel('Z Label')
        ax.set_ylabel('X Label')
        ax.set_zlabel('Y Label')
        plt.savefig(f'./vis/{idx}.jpg')    