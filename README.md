# Joint2Human: High-quality 3D Human Generation via Compact Spherical Embedding of 3D Joints (CVPR 2024)

### [Project Page](https://cic.tju.edu.cn/faculty/likun/projects/Joint2Human/index.html) | [Paper](https://arxiv.org/pdf/2312.08591.pdf) 

This is the official code for the CVPR 2024 paper "Joint2Human: High-quality 3D Human Generation via Compact Spherical Embedding of 3D Joints".


## Installation
Tested GPUs: A100, RTX4090
```
conda create -n j2h python=3.8
conda activate j2h
pip install -r requirements.txt
```

## Data processing
You can get the THUman2.1 Dataset from here [link](https://github.com/ytrock/THuman2.0-Dataset). 

1. FOF for human scan
```
python data/orth_mpi_obj.py
```
2. Compact spherical embedding of 3D joints for the SMPL data paired with thuman2.0.
```
python data/orth_joint_mpi.py
```
3. IUV maps
```
python data/render_iuv.py
```

## Training 

### Train the autoencoder for FOF
We used the code from [latent-diffusion](https://github.com/CompVis/latent-diffusion) to compress the FOF from [512,512,32] to [128,128,8].
### Train the Diffusion Model
```
python -m torch.distributed.launch --nproc_per_node=8 train.py 
```
## Citation
If you find this work useful for your research, please use the following BibTeX entry. 


```
@inproceedings{Joint2Human,
  author = {Muxin Zhang and Qiao Feng and Zhuo Su and Chao Wen and Zhou Xue and Kun Li},
  title = {Joint2Human: High-quality 3D Human Generation via Compact Spherical Embedding of 3D Joints},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```



