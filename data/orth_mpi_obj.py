import cyminiball as miniball
import numpy as np
from numba import jit
from numba.np.extensions import cross2d

import os
import multiprocessing

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

    # v = v.astype(np.float64) 
    # C,r2 = miniball.compute(v)
    # v = (v-C[:3])/np.sqrt(r2)
    return v,f

@jit(nopython=True)
def get_mpi(v, f, res):

    v[:,1] *= -1
    v[:,:2] = (v[:,:2]+1)*(res/2) - 0.5
    mpi = []
    for fid in f:
        pts = v[fid]
        iMax = int(np.ceil(np.max(pts[:,0])))
        iMax = min(res, iMax)
        iMin = int(np.ceil(np.min(pts[:,0])))
        iMin = max(0, iMin)
        jMax = int(np.ceil(np.max(pts[:,1])))
        jMax = min(res, jMax)
        jMin = int(np.ceil(np.min(pts[:,1])))
        jMin = max(0, jMin)
        for i in range(iMin, iMax):
            for j in range(jMin, jMax):
                p = np.array([i,j])
                w2 = cross2d(pts[1,:2] - pts[0,:2], p - pts[0,:2])
                w0 = cross2d(pts[2,:2] - pts[1,:2], p - pts[1,:2])
                w1 = cross2d(pts[0,:2] - pts[2,:2], p - pts[2,:2])
                ss = w0+w1+w2
                if ss==0:
                    continue
                elif ss>0:
                    if w0>=0 and w1>=0 and w2>=0:
                        mpi.append((j*res+i, (w0*pts[0,2]+w1*pts[1,2]+w2*pts[2,2])/ss, 0))
                elif ss<0:
                    if w0<=0 and w1<=0 and w2<=0:
                        mpi.append((j*res+i, (w0*pts[0,2]+w1*pts[1,2]+w2*pts[2,2])/ss, 1))
    mpi = sorted(mpi)
    pos = []
    ind = []
    val = []
    pre = 0
    cnt = 0
    last = -1
    while pre < len(mpi):
        while mpi[pre][2]==1 and pre<len(mpi):
            pre+=1
        if pre>=len(mpi):
            break
        nxt = pre+1

        flag = False
        while True:
            if nxt >= len(mpi) or mpi[nxt][0] != mpi[pre][0]:
                flag=True
                break
            if mpi[nxt][2]==1 and (nxt+1 >= len(mpi) or mpi[nxt+1][0] != mpi[pre][0] or mpi[nxt+1][2]==0):
                flag=False
                break
            nxt += 1

        if flag:
            pre = nxt
        else:
            if mpi[pre][0]!=last:
                pos.append(mpi[pre][0])
                ind.append(cnt)
                last = mpi[pre][0]
            val.append(mpi[pre][1])
            val.append(mpi[nxt][1])
            cnt+=2
            pre = nxt+1
    pos = np.array(pos, dtype=np.uint32)
    val = np.array(val, dtype=np.float32)
    ind = np.array(ind, dtype=np.uint32)
    return pos, ind, val

def work(d, src, tar, res):
    R = []
    for i in range(256): # 512 
        y  = np.pi*i/128 # 256
        sy = np.sin(y)
        cy = np.cos(y)
        r = np.array([  [ cy, 0.0,  sy],
                        [0.0, 1.0, 0.0],
                        [-sy, 0.0,  cy],] )
        R.append(r)

    obj_path = os.path.join(src, d, "smpl.obj")
    v,f = read_obj( obj_path)
    os.makedirs(os.path.join(tar, d), exist_ok=True)
    tmp = len(os.listdir(os.path.join(tar, d)))
    tmp = max(0,tmp-1)
    print(tmp)
    for view in range(tmp,256): 
        pos, ind, val = get_mpi(np.matmul(v, R[view].T), f, res)
        np.savez_compressed("%s/%s/%03d.npz" %(tar, d, view), pos=pos, ind=ind, val=val)

if __name__=="__main__":

    res = 512 
    src = "./data/thuman"  
    tar = "./data/mpi"   
    dd = os.listdir('./data/thuman')
    pool = multiprocessing.Pool(processes = 32)
    for d in dd:
        pool.apply_async(work, (d, src, tar, res,))    
    pool.close()
    pool.join()
    