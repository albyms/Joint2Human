import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .fof2occ import cos_occ
from utils.get_smpl_joint_ceof import smpl_joint_ooc
from PIL import Image

def preprocess_input(x):
    x /= 255
    x -= 0.5
    x /= 0.5
    return x

def postprocess_output(x):
    x *= 0.5
    x += 0.5
    x *= 255
    return x

def get_feature(npz_path):
    mpi = np.load(npz_path)
    pos = mpi['pos']
    ind = mpi['ind']
    val = mpi['val']

    feature = cos_occ(pos, ind, val, 64)
    feature = torch.from_numpy(feature).permute(2,0,1)
    return feature #[32,512,512]

def get_joint_feature(npz_path):
    mpi = np.load(npz_path)
    pos = mpi['pos']
    val = mpi['val']
    feature = smpl_joint_ooc(pos,val)
    feature = torch.from_numpy(feature).permute(2,0,1)
    return feature #[192,512,512] 

def get_iuv_feature(iuv_path):
    iuv = Image.open(iuv_path)
    iuv = iuv.convert("RGB")
    iuv = np.array(iuv, dtype=np.float32)
    
    iuv = np.transpose(iuv, (2, 0, 1))
    iuv= torch.from_numpy(iuv)[None]
    iuv = preprocess_input(iuv)
    return iuv[0]

class DdpmDataset(torch.utils.data.Dataset): 
    def __init__(self, name_list = "./data/train.txt") -> None:
        super().__init__()
        self.base_human_path = "./data/mpi"
        self.base_smpl_path  = "./data/joint-mpi"
        self.base_iuv_path = "./data/iuv"

        with open(name_list, "r") as f:
            self.name_list = f.read().split()

    def __len__(self):
        return len(self.name_list) * 32

    def __getitem__(self, index):
        pid = index // 32
        vid = index %  32
        svid = str(vid*8).zfill(3)
        rvid = str(vid).zfill(3)
        
        h_name = self.name_list[pid] + "/"+ svid+ ".npz"
        h_path = os.path.join(self.base_human_path,h_name)

        s_name = self.name_list[pid] + "/"+ svid + ".npz"
        s_path = os.path.join(self.base_smpl_path,s_name)

        iuv_name = self.name_list[pid] + "/"+ rvid + ".png"
        iuv_path = os.path.join(self.base_iuv_path,iuv_name)

        fof_h = get_feature(h_path)
        fof_s = get_joint_feature(s_path)
        fof_iuv = get_iuv_feature(iuv_path)
        return fof_h,fof_s,fof_iuv 

def get_dataloader(batch_size = 4):
    return torch.utils.data.DataLoader( DdpmDataset(),
                                        batch_size=batch_size,
                                        num_workers=12,
                                        shuffle=True,
                                        drop_last=True)

