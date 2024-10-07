import datetime,time
import os,sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)

from utils import logger

from utils.train_lfof_joint_dataset import DdpmDataset,get_dataloader
from utils.utils_tool import get_lr_scheduler, set_optimizer_lr, show_config
from utils.utils_fit_m import fit_one_epoch

from omegaconf import OmegaConf
from nets.autoencoderKL import AutoencoderKL


os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
if __name__ == "__main__":

    logger.configure(dir="./logs",info="xxxx") 
    
    Cuda = True
    distributed = True
    fp16 = False

    diffusion_model_path = ""
    channel = 128
    schedule = "linear"
    num_timesteps = 1000
    schedule_low = 1e-4
    schedule_high = 0.02
  
    input_shape = (128, 128)
    Init_Epoch = 222
    Epoch = 1000
    batch_size = 32
    Init_lr = 4e-5
    Min_lr = Init_lr * 0.5
    
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = "step"
    log_period = 200
    save_period = 1

    save_dir = "xxx"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    
    num_workers = 12
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache() 
    
    if schedule == "cosine":
        betas = generate_cosine_schedule(num_timesteps)
    else:
        betas = generate_linear_schedule(
            num_timesteps,
            schedule_low * 1000 / num_timesteps,
            schedule_high * 1000 / num_timesteps,
        )

    diffusion_model = GaussianDiffusion(UNet(8, channel), input_shape, 8, betas=betas)

    if diffusion_model_path != '':
        model_dict = diffusion_model.state_dict()
        pretrained_dict = torch.load(diffusion_model_path, map_location=torch.device('cpu'))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        diffusion_model.load_state_dict(model_dict)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    diffusion_model_train = diffusion_model.train()

    # 处理 pre_model 
    config_pos = "./configs/train-m.yaml" 
    config = OmegaConf.load(config_pos)

    weight_path = "xxxx/vae.pth"
    pre_model = AutoencoderKL(**config.model.get("params", dict()))
    pre_model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    pre_model_eval = pre_model.eval()

    diffusion_model_train = diffusion_model_train.cuda(local_rank)
    pre_model_eval = pre_model_eval.cuda(local_rank)
    if distributed:
        diffusion_model_train = torch.nn.parallel.DistributedDataParallel(diffusion_model_train,
                                                                            device_ids=[local_rank],
                                                                            find_unused_parameters=True)
        pre_model_eval = torch.nn.parallel.DistributedDataParallel(pre_model_eval,
                                                                    device_ids=[local_rank],
                                                                    find_unused_parameters=True)
    train_dataset = DdpmDataset()
    num_train = len(train_dataset)

    logger.log(f'The dataset size = {num_train}')

    if local_rank == 0:
        show_config(
            input_shape=input_shape, Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size, \
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type, \
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train
        )
    if True:
        optimizer = {
            'adam': optim.Adam(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                               weight_decay=weight_decay),
            'adamw': optim.AdamW(diffusion_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999),
                                 weight_decay=weight_decay),
        }[optimizer_type]
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)
        epoch_step = num_train // batch_size
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            shuffle = True
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, sampler=train_sampler)
        for epoch in range(Init_Epoch, Epoch):

            if distributed:
                gen.sampler.set_epoch(epoch)
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(pre_model_eval,
                          diffusion_model_train, 
                          diffusion_model,optimizer,
                          epoch,epoch_step,
                          gen, 
                          Epoch, 
                          Cuda, 
                          fp16, 
                          scaler, 
                          log_period,
                          save_period, 
                          save_dir, 
                          local_rank,
                          puncond=0.1)

            if distributed:
                dist.barrier()