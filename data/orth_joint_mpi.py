import numpy as np
from numba import jit
from numba.np.extensions import cross2d

import os
import multiprocessing
import math
import torch


data_regressor = np.load('./J_regressor.npz')
J_regressor = data_regressor['data']
def read_obj(filename):
    v = []
    f = []
    with open(filename, "r") as fi:
        lines = fi.readlines()
        for line in lines:
            tmp = line.split()

            if len(tmp) > 0 and tmp[0] == "v":
                v.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
            elif len(tmp) > 0 and tmp[0] =="f":
                ttt = [ tmp[1].split("/"),
                        tmp[2].split("/"),
                        tmp[3].split("/")]
                f.append([  int(ttt[0][0])-1, 
                            int(ttt[1][0])-1, 
                            int(ttt[2][0])-1])
    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.uint32)

    v = v.astype(np.float64) 
    return v,f

def get_eudis(x,center):
    dx = x[0]-center[0]
    dy = x[1]-center[1]
    dz = x[2]-center[2]
    return dx*dx + dy*dy + dz*dz

def get_one_smpl_mpi(res=512,R=10,num=8):    

    v = np.array([[0.111659,0.06592,0.10938],],dtype=np.float32) 
    v[:,1] *= -1
    v[:,:2] = (v[:,:2]+1)*(res/2) - 0.5

    iMax = int(np.ceil(v[0,0]+R))
    iMax = min(res,iMax)
    iMin = int(np.ceil(v[0,0]-R))
    iMin = max(0,iMin)

    jMax = int(np.ceil(v[0,1]+R))
    jMax = min(res,jMax)
    jMin = int(np.ceil(v[0,1]-R))
    jMin = max(0,jMin)

    ceof = np.zeros((512, 512, num), dtype=np.float32)
    ktmp = np.zeros(num, dtype=np.float32)

    for k in range(1,num,1):
        ktmp[k] = k*np.pi*0.5

    for i in range(iMin,iMax):
        for j in range(jMin,jMax):
            st = -1
            z_mirror = (v[0,2]+1)*(res/2)-0.5
            for z in range(0,512):
                if(get_eudis([i,j,z],[v[0,0],v[0,1],z_mirror]) <= R*R):
                    st = z
                    break
            if(st!=-1):
                ed = math.ceil(2 * z_mirror - st)
                pos = np.arange(st,ed+1)
                pos = (pos+0.5)*2/res 
                for k in range(0,len(pos)-1,2):
                    ceof[i,j,0] += (pos[k+1]-pos[k])
                    for m in range(1,num,1):
                        ceof[i, j, m] += (np.sin(ktmp[m]*pos[k+1])-np.sin(ktmp[m]*pos[k])) / ktmp[m]
                
    ceof = torch.from_numpy(ceof).permute(2,0,1)
    return ceof

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
                else:
                    val.append(st)

    pos = np.array(pos, dtype=np.int16)
    val = np.array(val, dtype=np.int16)
    return pos,val


def work(d, src, tar, res):
    R = []
    for i in range(256): 
        y  = np.pi*i/128 
        sy = np.sin(y)
        cy = np.cos(y)
        r = np.array([  [ cy, 0.0,  sy],
                        [0.0, 1.0, 0.0],
                        [-sy, 0.0,  cy],] )
        R.append(r)

    obj_path = os.path.join(src,d,'smpl.obj')
    v,f = read_obj(obj_path)
    os.makedirs(os.path.join(tar, d), exist_ok=True)
    tmp = len(os.listdir(os.path.join(tar, d)))
    tmp = max(0,tmp-1)
    print(tmp)
    for view in range(tmp,256): 
        pos,val = get_smpl_mpi(np.matmul(v, R[view].T),res=res)
        np.savez_compressed("%s/%s/%03d.npz" %(tar, d, view), pos=pos,val=val)

if __name__=="__main__":

    src = './data/smpl'
    tar = './data/joint-mpi'  

    res = 512 
    obj_list = os.listdir(src)

    pool = multiprocessing.Pool(processes = 128)
    for d in obj_list:
        pool.apply_async(work, (d, src, tar, res,))     
    pool.close()
    pool.join()
    