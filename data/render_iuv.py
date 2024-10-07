import numpy as np
from scipy.io import loadmat
import scipy.spatial.distance
from numba import cuda
import numba.cuda.libdevice as ld 
import cv2
from tqdm import tqdm
import os

class DensePoseMethods:
    def __init__(self):
        #
        ALP_UV = loadmat(os.path.join('./data/uv_data', 'UV_Processed.mat'))
        self.FaceIndices = np.array(ALP_UV['All_FaceIndices']).squeeze()
        self.FacesDensePose = ALP_UV['All_Faces'] - 1
        self.U_norm = ALP_UV['All_U_norm'].squeeze()
        self.V_norm = ALP_UV['All_V_norm'].squeeze()
        self.All_vertices = ALP_UV['All_vertices'][0]
        ## Info to compute symmetries.
        self.SemanticMaskSymmetries = [0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 14]
        self.Index_Symmetry_List = [1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24,
                                    23];
        UV_symmetry_filename = os.path.join('./uv_data', 'UV_symmetry_transforms.mat')
        self.UV_symmetry_transformations = loadmat(UV_symmetry_filename)

    def get_symmetric_densepose(self, I, U, V, x, y, Mask):
        ### This is a function to get the mirror symmetric UV labels.
        Labels_sym = np.zeros(I.shape)
        U_sym = np.zeros(U.shape)
        V_sym = np.zeros(V.shape)
        ###
        for i in (range(24)):
            if i + 1 in I:
                Labels_sym[I == (i + 1)] = self.Index_Symmetry_List[i]
                jj = np.where(I == (i + 1))
                ###
                U_loc = (U[jj] * 255).astype(np.int32)
                V_loc = (V[jj] * 255).astype(np.int32)
                ###
                V_sym[jj] = self.UV_symmetry_transformations['V_transforms'][0, i][V_loc, U_loc]
                U_sym[jj] = self.UV_symmetry_transformations['U_transforms'][0, i][V_loc, U_loc]
        ##
        Mask_flip = np.fliplr(Mask)
        Mask_flipped = np.zeros(Mask.shape)
        #
        for i in (range(14)):
            Mask_flipped[Mask_flip == (i + 1)] = self.SemanticMaskSymmetries[i + 1]
        #
        [y_max, x_max] = Mask_flip.shape
        y_sym = y
        x_sym = x_max - x
        #
        return Labels_sym, U_sym, V_sym, x_sym, y_sym, Mask_flipped

    def barycentric_coordinates_exists(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        if (np.dot(vCrossW, vCrossU) < 0):
            return False;
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        if (np.dot(uCrossW, uCrossV) < 0):
            return False;
        #
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        #
        return ((r <= 1) & (t <= 1) & (r + t <= 1))

    def barycentric_coordinates(self, P0, P1, P2, P):
        u = P1 - P0
        v = P2 - P0
        w = P - P0
        #
        vCrossW = np.cross(v, w)
        vCrossU = np.cross(v, u)
        #
        uCrossW = np.cross(u, w)
        uCrossV = np.cross(u, v)
        #
        denom = np.sqrt((uCrossV ** 2).sum())
        r = np.sqrt((vCrossW ** 2).sum()) / denom
        t = np.sqrt((uCrossW ** 2).sum()) / denom
        #
        return (1 - (r + t), r, t)

    def IUV2FBC(self, I_point, U_point, V_point):
        P = [U_point, V_point, 0]
        FaceIndicesNow = np.where(self.FaceIndices == I_point)
        FacesNow = self.FacesDensePose[FaceIndicesNow]
        #
        P_0 = np.vstack((self.U_norm[FacesNow][:, 0], self.V_norm[FacesNow][:, 0],
                         np.zeros(self.U_norm[FacesNow][:, 0].shape))).transpose()
        P_1 = np.vstack((self.U_norm[FacesNow][:, 1], self.V_norm[FacesNow][:, 1],
                         np.zeros(self.U_norm[FacesNow][:, 1].shape))).transpose()
        P_2 = np.vstack((self.U_norm[FacesNow][:, 2], self.V_norm[FacesNow][:, 2],
                         np.zeros(self.U_norm[FacesNow][:, 2].shape))).transpose()
        #

        for i, [P0, P1, P2] in enumerate(zip(P_0, P_1, P_2)):
            if (self.barycentric_coordinates_exists(P0, P1, P2, P)):
                [bc1, bc2, bc3] = self.barycentric_coordinates(P0, P1, P2, P)
                return (FaceIndicesNow[0][i], bc1, bc2, bc3)
        #
        # If the found UV is not inside any faces, select the vertex that is closest!
        #
        D1 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_0[:, 0:2]).squeeze()
        D2 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_1[:, 0:2]).squeeze()
        D3 = scipy.spatial.distance.cdist(np.array([U_point, V_point])[np.newaxis, :], P_2[:, 0:2]).squeeze()
        #
        minD1 = D1.min()
        minD2 = D2.min()
        minD3 = D3.min()
        #
        if ((minD1 < minD2) & (minD1 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D1)], 1., 0., 0.)
        elif ((minD2 < minD1) & (minD2 < minD3)):
            return (FaceIndicesNow[0][np.argmin(D2)], 0., 1., 0.)
        else:
            return (FaceIndicesNow[0][np.argmin(D3)], 0., 0., 1.)

    def FBC2PointOnSurface(self, FaceIndex, bc1, bc2, bc3, Vertices):
        ##
        Vert_indices = self.All_vertices[self.FacesDensePose[FaceIndex]] - 1
        ##
        p = Vertices[Vert_indices[0], :] * bc1 + \
            Vertices[Vert_indices[1], :] * bc2 + \
            Vertices[Vert_indices[2], :] * bc3
        ##
        return (p)
    
@cuda.jit
def get_vn_0(v, f, vn):
    fid = cuda.grid(1)
    if fid >= f.shape[0]: return
    vid = f[fid]
    p1 = v[vid[0]]
    p2 = v[vid[1]]
    p3 = v[vid[2]]
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    z1 = p2[2] - p1[2]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    z2 = p3[2] - p1[2]
    nx = y1*z2-z1*y2
    ny = z1*x2-x1*z2
    nz = x1*y2-y1*x2
    for i in range(3):
        cuda.atomic.add(vn[vid[i]],0,nx)
        cuda.atomic.add(vn[vid[i]],1,ny)
        cuda.atomic.add(vn[vid[i]],2,nz)
@cuda.jit
def get_vn_1(vn):
    vid = cuda.grid(1)
    if vid >= vn.shape[0]: return
    n = vn[vid]
    tmp = ld.fsqrt_rn(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
    tmp = ld.fmax(tmp,1e-6)
    for i in range(3):
        n[i] /= tmp

@cuda.jit
def get_trans(v,vn,tmp_v,tmp_vn,r,res):
    vid = cuda.grid(1)
    if vid >= vn.shape[0]: return
    tmp_vn[vid,0] = vn[vid,0]*r[0,0] + vn[vid,1]*r[0,1] + vn[vid,2]*r[0,2]
    tmp_vn[vid,1] = vn[vid,0]*r[1,0] + vn[vid,1]*r[1,1] + vn[vid,2]*r[1,2]
    tmp_vn[vid,2] = vn[vid,0]*r[2,0] + vn[vid,1]*r[2,1] + vn[vid,2]*r[2,2]

    x = v[vid,0]*r[0,0] + v[vid,1]*r[0,1] + v[vid,2]*r[0,2]
    y = v[vid,0]*r[1,0] + v[vid,1]*r[1,1] + v[vid,2]*r[1,2]
    z = v[vid,0]*r[2,0] + v[vid,1]*r[2,1] + v[vid,2]*r[2,2]

    tmp_v[vid,0] = (1+x)*(res/2) - 0.5
    tmp_v[vid,1] = (1-y)*(res/2) - 0.5
    tmp_v[vid,2] = z


@cuda.jit
def get_depth(v, f, depth, cnt):
    fid = cuda.grid(1)
    lim = f.shape[0]
    res = depth.shape[0]
    if fid >= lim: return

    vid = f[fid]
    p1 = v[vid[0]]
    p2 = v[vid[1]]
    p3 = v[vid[2]]
    
    iMax = ld.min(ld.ceilf(max(p1[0],p2[0],p3[0])), res) # x+1
    jMax = ld.min(ld.ceilf(max(p1[1],p2[1],p3[1])), res) # x+1
    iMin = ld.max(ld.ceilf(min(p1[0],p2[0],p3[0])), 0)
    jMin = ld.max(ld.ceilf(min(p1[1],p2[1],p3[1])), 0)
    
    for j in range(jMin, jMax):
        for i in range(iMin, iMax):
            w3 = (p2[0]-p1[0])*(j-p1[1]) - (p2[1]-p1[1])*(i-p1[0])
            w1 = (p3[0]-p2[0])*(j-p2[1]) - (p3[1]-p2[1])*(i-p2[0])
            w2 = (p1[0]-p3[0])*(j-p3[1]) - (p1[1]-p3[1])*(i-p3[0])
            ss = w1+w2+w3
            if ss>0:
                if w1>=0 and w2>=0 and w3>=0:
                    cuda.atomic.add(cnt[j,i],0,1)
                    d = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss
                    cuda.atomic.max(depth[j,i],0,d)
            elif ss<0:
                if w1<=0 and w2<=0 and w3<=0:
                    cuda.atomic.add(cnt[j,i],0,1)
                    d = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss
                    cuda.atomic.max(depth[j,i],0,d)

@cuda.jit
def get_normal(v, f, vn, depth, normal):
    fid = cuda.grid(1)
    lim = f.shape[0]
    res = depth.shape[0]
    if fid >= lim: return

    vid = f[fid]
    p1 = v[vid[0]]
    p2 = v[vid[1]]
    p3 = v[vid[2]]
    
    iMax = ld.min(ld.ceilf(max(p1[0],p2[0],p3[0])), res) # x+1
    jMax = ld.min(ld.ceilf(max(p1[1],p2[1],p3[1])), res) # x+1
    iMin = ld.max(ld.ceilf(min(p1[0],p2[0],p3[0])), 0)
    jMin = ld.max(ld.ceilf(min(p1[1],p2[1],p3[1])), 0)
    
    for j in range(jMin, jMax):
        for i in range(iMin, iMax):
            w3 = (p2[0]-p1[0])*(j-p1[1]) - (p2[1]-p1[1])*(i-p1[0])
            w1 = (p3[0]-p2[0])*(j-p2[1]) - (p3[1]-p2[1])*(i-p2[0])
            w2 = (p1[0]-p3[0])*(j-p3[1]) - (p1[1]-p3[1])*(i-p3[0])
            ss = w1+w2+w3
            if ss>0:
                if w1>=0 and w2>=0 and w3>=0:
                    d = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss
                    if ld.double2float_rn(d)==depth[j,i,0]:
                        vn1 = vn[vid[0]]
                        vn2 = vn[vid[1]]
                        vn3 = vn[vid[2]]
                        n1 = (w1*vn1[0]+w2*vn2[0]+w3*vn3[0])/ss
                        n2 = (w1*vn1[1]+w2*vn2[1]+w3*vn3[1])/ss
                        n3 = (w1*vn1[2]+w2*vn2[2]+w3*vn3[2])/ss
                        nn = ld.dsqrt_rn(n1*n1 + n2*n2 + n3*n3)
                        normal[j,i,0] = n1/nn
                        normal[j,i,1] = n2/nn
                        normal[j,i,2] = n3/nn                   
            elif ss<0:
                if w1<=0 and w2<=0 and w3<=0:
                    d = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss
                    if ld.double2float_rn(d)==depth[j,i,0]:
                        vn1 = vn[vid[0]]
                        vn2 = vn[vid[1]]
                        vn3 = vn[vid[2]]
                        n1 = (w1*vn1[0]+w2*vn2[0]+w3*vn3[0])/ss
                        n2 = (w1*vn1[1]+w2*vn2[1]+w3*vn3[1])/ss
                        n3 = (w1*vn1[2]+w2*vn2[2]+w3*vn3[2])/ss
                        nn = ld.dsqrt_rn(n1*n1 + n2*n2 + n3*n3)
                        normal[j,i,0] = n1/nn
                        normal[j,i,1] = n2/nn
                        normal[j,i,2] = n3/nn

@cuda.jit
def get_iuv(v, f, vt, depth, iuvmap):
    fid = cuda.grid(1)
    lim = f.shape[0]
    res = depth.shape[0]
    if fid >= lim: return

    vid = f[fid]
    p1 = v[vid[0]]
    p2 = v[vid[1]]
    p3 = v[vid[2]]
    
    iMax = ld.min(ld.ceilf(max(p1[0],p2[0],p3[0])), res) # x+1
    jMax = ld.min(ld.ceilf(max(p1[1],p2[1],p3[1])), res) # x+1
    iMin = ld.max(ld.ceilf(min(p1[0],p2[0],p3[0])), 0)
    jMin = ld.max(ld.ceilf(min(p1[1],p2[1],p3[1])), 0)
    
    for j in range(jMin, jMax):
        for i in range(iMin, iMax):
            w3 = (p2[0]-p1[0])*(j-p1[1]) - (p2[1]-p1[1])*(i-p1[0])
            w1 = (p3[0]-p2[0])*(j-p2[1]) - (p3[1]-p2[1])*(i-p2[0])
            w2 = (p1[0]-p3[0])*(j-p3[1]) - (p1[1]-p3[1])*(i-p3[0])
            ss = w1+w2+w3
            if ss>0:
                if w1>=0 and w2>=0 and w3>=0:
                    d = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss
                    if ld.double2float_rn(d)==depth[j,i,0]:
                        vt1 = vt[vid[0]]
                        vt2 = vt[vid[1]]
                        vt3 = vt[vid[2]]
                        # 记录iuv
                        n1 = (w1*vt1[0]+w2*vt2[0]+w3*vt3[0])/ss
                        n2 = (w1*vt1[1]+w2*vt2[1]+w3*vt3[1])/ss
                        iuvmap[j,i,0] = vid[3]/24
                        iuvmap[j,i,1] = n1
                        iuvmap[j,i,2] = n2                   
            elif ss<0:
                if w1<=0 and w2<=0 and w3<=0:
                    d = (w1*p1[2]+w2*p2[2]+w3*p3[2])/ss
                    if ld.double2float_rn(d)==depth[j,i,0]:
                        vt1 = vt[vid[0]]
                        vt2 = vt[vid[1]]
                        vt3 = vt[vid[2]]
                        # 记录iuv
                        n1 = (w1*vt1[0]+w2*vt2[0]+w3*vt3[0])/ss
                        n2 = (w1*vt1[1]+w2*vt2[1]+w3*vt3[1])/ss
                        iuvmap[j,i,0] = vid[3]/24
                        iuvmap[j,i,1] = n1
                        iuvmap[j,i,2] = n2

def load_notex_obj(filename):
    v=[]
    f=[]
    with open(filename, "r") as fi:
        lines = fi.readlines()
        for line in lines:
            tmp = line.split()
            if tmp[0] == "v":
                v.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
            elif tmp[0] =="f":
                f.append([float(tmp[1])-1, float(tmp[2])-1, float(tmp[3])-1])
    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.uint32)
    return v,f

def work(name, res, vnum):
    path_output = os.path.join("./data/iuv",name)
    DP = DensePoseMethods()
    vert_mapping = DP.All_vertices.astype('int32') - 1
    
    v,f = load_notex_obj(os.path.join(f'./data/smpl/{name}','smpl.obj'))
    v = v[vert_mapping,:]
    f = np.hstack([DP.FacesDensePose,DP.FaceIndices.reshape((-1,1))])
    vt = np.vstack([DP.U_norm,DP.V_norm]).T
    
    vt_cuda = cuda.to_device(vt)
    v_cuda = cuda.to_device(v)
    f_cuda = cuda.to_device(f)
    vn_cuda = cuda.device_array_like(v_cuda)

    tmp_v_cuda = cuda.device_array_like(v_cuda)
    tmp_vn_cuda = cuda.device_array_like(vn_cuda)
    
    depth = np.ones((512,512,1), dtype=np.float32)*-1
    cnt = np.zeros((512,512,1), dtype=np.int32)
    iuvmap = np.zeros((512,512,3), dtype=np.float32)
    os.makedirs(path_output,exist_ok=True)
    for i in range(vnum):
        # 1. Rotation & Transform
        y = np.pi*i/(vnum//2)
        sy = np.sin(y)
        cy = np.cos(y)
        r = np.array([  [ cy, 0.0,  sy],
                        [0.0, 1.0, 0.0],
                        [-sy, 0.0,  cy],], dtype=np.float32)
        r = cuda.to_device(r)
        get_trans[(v.shape[0]+63)//64,64](v_cuda, vn_cuda, tmp_v_cuda, tmp_vn_cuda, r, res)
    
        # 2. get Depth
        depth_cuda = cuda.to_device(depth)
        cnt_cuda = cuda.to_device(cnt)
        get_depth[(f.shape[0]+63)//64,64](tmp_v_cuda,f_cuda,depth_cuda,cnt_cuda)

        # 4. Calculate iuvmap
        iuvmap_cuda = cuda.to_device(iuvmap)
        get_iuv[(f.shape[0]+63)//64,64](tmp_v_cuda, f_cuda, vt_cuda, depth_cuda, iuvmap_cuda)
        tmp = iuvmap_cuda.copy_to_host()
        tmp = tmp*255
        
        cv2.imwrite(os.path.join(path_output,"%03d.png"%i), tmp)

if __name__ == "__main__":
    name_list = os.listdir('./data/thuman')
    for file in tqdm(name_list): 
        work(file, 512, 256)
