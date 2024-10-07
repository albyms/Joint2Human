import torch
import numpy as np
from skimage.measure import marching_cubes
from numba import njit
import tqdm
from mc import mc
@njit
def cos_occ(pos, ind, val, num):
    ceof = np.zeros((512, 512, num), dtype=np.float32)
    ktmp = np.zeros(num, dtype=np.float32)
    for k in range(1,num,1):
        ktmp[k] = k*np.pi*0.5
    val = val+1
    ind = np.append(ind, len(ind))
    for i in range(len(pos)):
        xx = pos[i]>>9
        yy = pos[i]&511
        for j in range(ind[i], ind[i+1], 2):
            ceof[xx, yy, 0] += (val[j+1]-val[j])
            for k in range(1, num, 1): 
                ceof[xx, yy, k] += (np.sin(ktmp[k]*val[j+1])-np.sin(ktmp[k]*val[j])) / ktmp[k]
    return ceof

class Recon():
    def __init__(self, device, dim, mat = None) -> None:
        self.device = device
        if mat == None:
            z = (torch.arange(512, dtype=torch.float32, device=device)+0.5)/512
            z = torch.arange(dim, dtype=torch.float32, device=device).view(1, dim) * z.view(512, 1) * np.pi
            self.z = torch.cos(z)
            self.z[:,0] = 0.5
        else:
            mat = mat.to(device)
            z = (torch.arange(512, dtype=torch.float32, device=device)+0.5)/512
            z = torch.arange(512, dtype=torch.float32, device=device).view(1, 512) * z.view(512, 1) * np.pi
            z = torch.cos(z)
            z[:,0] = 0.5
            self.z = torch.matmul(z, mat)

    def decode(self, ceof):
        with torch.no_grad():
            res = torch.einsum("dc, chw -> dhw", self.z, ceof)
            # v, f, _, _ = marching_cubes(res.cpu().numpy(), level = 0.5)
            v, f, _, _ = marching_cubes(res.cpu().numpy())
        v += 0.5
        v = v/256 - 1
        v[:,1] *= -1
        vv = np.zeros_like(v)
        vv[:,0] = v[:,2]
        vv[:,1] = v[:,1]
        vv[:,2] = v[:,0]

        ff = np.zeros_like(f)
        ff[:,0] = f[:,0]
        ff[:,1] = f[:,2]
        ff[:,2] = f[:,1]
        return vv,ff
    
    def decode_pro(self, ceof):
        with torch.no_grad():
            res = torch.einsum("dc, chw -> dhw", self.z, ceof)
            v, f, _, _ = marching_cubes(res.cpu().numpy(), level = 0.5)
            
        v += 0.5
        v = v/256 - 1
        v[:,1] *= -1
        vv = np.zeros_like(v)
        vv[:,0] = v[:,2]
        vv[:,1] = v[:,1]
        vv[:,2] = v[:,0]

        ff = np.zeros_like(f)
        ff[:,0] = f[:,0]
        ff[:,1] = f[:,2]
        ff[:,2] = f[:,1]
        # return vv, ff, res.cpu().numpy()
        return vv, ff, res.cpu()
    
    def decode_fq(self, ceof):
        with torch.no_grad():
            res = torch.einsum("dc, chw -> dhw", self.z, ceof)

            v,f = mc(res.cpu().numpy(),threshold = 0.5)
        v += 0.5
        v = v/256 - 1

        return v,f

    def get_vf(self,res):
        v, f, _, _ = marching_cubes(res.cpu().numpy(), level=0.5)
        v += 0.5
        v = v / 256 - 1
        v[:, 1] *= -1
        vv = np.zeros_like(v)
        vv[:, 0] = v[:, 2]
        vv[:, 1] = v[:, 1]
        vv[:, 2] = v[:, 0]

        ff = np.zeros_like(f)
        ff[:, 0] = f[:, 0]
        ff[:, 1] = f[:, 2]
        ff[:, 2] = f[:, 1]
        return vv, ff


if __name__ == "__main__":
    base = "xxx"
    a = Recon("cpu",32)
    for i in tqdm.tqdm(range(0,512,64)):
        t = np.load(base+"%03d.npz"%i)
        fof = cos_occ(t["pos"], t["ind"], t["val"], 32)

        fof = torch.from_numpy(fof).permute(2,0,1)
        v,f = a.decode(fof)
        with open("%03d.obj"%i, "w") as fi:
            for vv in v:
                fi.write("v %f %f %f\n" % (vv[0],vv[1],vv[2]))
            for ff in f+1:
                fi.write("f %d %d %d\n" % (ff[0],ff[1],ff[2]))
