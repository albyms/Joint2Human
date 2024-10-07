from multiprocessing import pool
import cyminiball as miniball
import numpy as np
from numba import jit
from numba.np.extensions import cross2d
from utils.fof2occ import cos_occ,Recon

import os
import multiprocessing
import math
import torch
import trimesh

def clean_mesh(mesh_path):

    mesh_path = 'xxx'
    mesh = trimesh.load(mesh_path)
    dis_mesh = mesh.split()
    max_mesh_vets = 0
    ans = dis_mesh[0]
    for item in dis_mesh:
        if item.vertices.shape[0] > max_mesh_vets:
            ans = item
            max_mesh_vets = item.vertices.shape[0]

    obj_save_path = mesh_path[:-4] +'_clean.obj'
    v = ans.vertices
    f = ans.faces
    v = v *100 + 100
    with open(obj_save_path,"w") as mf:
        for i in v:
            mf.write("v %f %f %f\n" % (i[0], i[1], i[2]))
        for i in f:
            mf.write("f %d %d %d\n" % (i[0] + 1, i[1] + 1, i[2] + 1))
    
    

def get_smpl_mpi(v,res=512,R=10,num=8,joint_cnt=24):    

    joint = np.matmul(J_regressor,v)
    pos = [] 
    val = [] 
    joint[:,1] *= -1
    joint[:,:2] = (joint[:,:2]+1)*(res/2) - 0.5

    for jdx in range(joint_cnt):
        v = joint[jdx]
        iMax = int(np.ceil(v[0]+R))
        iMax = min(res,iMax)
        iMin = int(np.ceil(v[0]-R))
        iMin = max(0,iMin)

        jMax = int(np.ceil(v[1]+R))
        jMax = min(res,jMax)
        jMin = int(np.ceil(v[1]-R))
        jMin = max(0,jMin)
        pos.append(np.array([iMin,iMax,jMin,jMax], dtype=np.uint32))

        for i in range(iMin,iMax):
            for j in range(jMin,jMax):
                st = -1
                z_mirror = (v[2]+1)*(res/2)-0.5
                for z in range(0,512):
                    if(get_eudis([i,j,z],[v[0],v[1],z_mirror]) <= R*R):
                        st = z
                        break
                
                val.append(st)
                if(st!=-1):
                    ed = math.ceil(2 * z_mirror - st)
                    val.append(ed)
                val.append(st) 


    pos = np.array(pos, dtype=np.int16)
    val = np.array(val, dtype=np.int16)
    return pos,val

def smpl_joint_ooc(pos,val,res=512,num=8):
    idx = 0
    res = 512
    num = 8
    ceof = np.zeros((512, 512, num*pos.shape[0]), dtype=np.float32)
    ktmp = np.zeros(num, dtype=np.float32)

    for k in range(1,num,1):
        ktmp[k] = k*np.pi*0.5
    for jdx in range(pos.shape[0]):
        iMin,iMax,jMin,jMax = pos[jdx]
        for i in range(iMin,iMax):
            for j in range(jMin,jMax):
                st = val[idx]             
                if(st == -1):
                    idx+=2
                    continue
                ed = val[idx+1]
                idx+=2
                ind = np.arange(st,ed+1)
                ind = (ind+0.5)*2/res 
                for k in range(0,len(ind)-1,2):
                    ceof[j,i,jdx*num] += (ind[k+1]-ind[k])
                    for m in range(1,num,1):
                        ceof[j,i,jdx*num+m] += (np.sin(ktmp[m]*ind[k+1])-np.sin(ktmp[m]*ind[k])) / ktmp[m]
    return ceof

if __name__ == "__main__":

    pool = os.listdir('xxx')
    for item in pool:
        npz_path = 'xxx'
        data = np.load(npz_path)

        pos = data['pos']
        val = data['val']

        idx = 0
        res = 512
        num = 8
        ceof = np.zeros((512, 512, num*pos.shape[0]), dtype=np.float32)
        ktmp = np.zeros(num, dtype=np.float32)

        for k in range(1,num,1):
            ktmp[k] = k*np.pi*0.5

        for jdx in range(pos.shape[0]):
            iMin,iMax,jMin,jMax = pos[jdx]
            for i in range(iMin,iMax):
                for j in range(jMin,jMax):
                    st = val[idx]             
                    if(st == -1):
                        idx+=2
                        continue
                    ed = val[idx+1]
                    idx+=2
                    ind = np.arange(st,ed+1)
                    ind = (ind+0.5)*2/res 
                    for k in range(0,len(ind)-1,2):
                        ceof[j,i,jdx*num] += (ind[k+1]-ind[k])
                        for m in range(1,num,1):
                            ceof[j,i,jdx*num+m] += (np.sin(ktmp[m]*ind[k+1])-np.sin(ktmp[m]*ind[k])) / ktmp[m]

        ceof = torch.from_numpy(ceof).permute(2,0,1)
        device = torch.device('cuda:0')
        e = Recon(device,8)
        t_ceof = torch.zeros(8,512,512).to(device) 
        ceof=ceof.to(device)

        for i in range(24):
            t_ceof += ceof[i*8:i*8+8,:,:]

        obj_save_path = os.path.join('xxx',f'{item}.obj')
        with torch.no_grad():
            v,f = e.decode(t_ceof)            
            with open(obj_save_path, "w") as mf:
                for i in v:
                    mf.write("v %f %f %f\n" % (i[0], i[1], i[2]))
                for i in f:
                    mf.write("f %d %d %d\n" % (i[0] + 1, i[1] + 1, i[2] + 1))
            clean_mesh(obj_save_path)
