import torch
import numpy as np
import os,sys
import time,datetime
from omegaconf import OmegaConf
from utils.fof2occ import cos_occ,Recon


def get_data_list(txt_path):
    with open(txt_path,'r') as f:
        test_list  = f.read().split()
    return test_list

def get_test_data(path):
    mpi = np.load(path)
    pos = mpi['pos']
    ind = mpi['ind']
    val = mpi['val']
    
    fof = cos_occ(pos, ind, val, 32)
    fof = torch.from_numpy(fof).permute(2,0,1)
    return fof

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
a = Recon(device,32)

test_fof_data_path = 'xxxx/000.npz'

data = np.load(test_fof_data_path)
fof_data = get_test_data(test_fof_data_path)[None]
fof_data = fof_data.to(device)
v,f = a.decode_fq(fof_data[0])

obj_save_path = 'xxx.obj'
with open(obj_save_path , "w") as mf:
    for i in v:
        mf.write("v %f %f %f\n" % (i[0], i[1], i[2]))
    for i in f:
        mf.write("f %d %d %d\n" % (i[0] + 1, i[1] + 1, i[2] + 1))
