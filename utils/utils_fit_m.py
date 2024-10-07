from ntpath import join
import os

import torch
import torch.distributed as dist
from tqdm import tqdm

from utils import logger
from utils.utils_tool import get_lr
import numpy as np
import torch.nn.functional as F

def fit_one_epoch_nocond(pre_model_eval,diffusion_model_train, diffusion_model, optimizer,epoch, epoch_step, 
                  gen, Epoch, cuda, fp16, scaler,log_period,save_period, save_dir, local_rank=0):
    total_loss = 0

    if local_rank == 0:
        logger.log('Start Train !!!')

    for iteration, images in enumerate(gen):
        if iteration >= epoch_step:
            break

        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                images = pre_model_eval(images)

        if not fp16:
            optimizer.zero_grad()
            diffusion_loss = torch.mean(diffusion_model_train(images))
            diffusion_loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            optimizer.zero_grad()
            with autocast():
                diffusion_loss = torch.mean(diffusion_model_train(images))

            scaler.scale(diffusion_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        diffusion_model.update_ema()

        total_loss += diffusion_loss.item()
        if local_rank == 0:
            if (iteration % log_period == 0): 
                logger.log(f'epoch:{epoch + 1},iteration: {iteration}/{epoch_step}, total_loss: {total_loss / (iteration + 1)}, lr: {get_lr(optimizer)}')

    total_loss = total_loss / epoch_step
    if local_rank == 0:
        logger.log('Epoch:' + str(epoch + 1) + '/' + str(Epoch) + ',Total_loss: %.4f ' % (total_loss))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(diffusion_model.state_dict(),
                       os.path.join(save_dir, 'Diffusion_Epoch%d-GLoss%.4f.pth' % (epoch + 1, total_loss)))

def fit_one_epoch(pre_model_eval,diffusion_model_train, diffusion_model, optimizer,epoch, epoch_step, 
                  gen, Epoch, cuda, fp16, scaler,log_period,save_period, save_dir, local_rank=0,puncond=0.1):
    total_loss = 0

    if local_rank == 0:
        logger.log('Start Train !!!')

    for iteration, [images,joint,iuv] in enumerate(gen):
        if iteration >= epoch_step:
            break
        b = images.shape[0]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                images = pre_model_eval(images)
                joint = joint.cuda(local_rank)
                iuv = iuv.cuda(local_rank)

        if not fp16:
            optimizer.zero_grad()
            diffusion_loss = torch.mean(diffusion_model_train(images,joint=joint,iuv=iuv))
            diffusion_loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            optimizer.zero_grad()
            with autocast():
                diffusion_loss = torch.mean(diffusion_model_train(images,joint=joint,iuv=iuv))

            scaler.scale(diffusion_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        diffusion_model.update_ema()

        total_loss += diffusion_loss.item()
        if local_rank == 0:
            if (iteration % log_period == 0):  
                logger.log(f'epoch:{epoch + 1},iteration: {iteration}/{epoch_step}, total_loss: {total_loss / (iteration + 1)}, lr: {get_lr(optimizer)}')

    total_loss = total_loss / epoch_step

    if local_rank == 0:
        logger.log('Epoch:' + str(epoch + 1) + '/' + str(Epoch) + ',Total_loss: %.4f ' % (total_loss))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(diffusion_model.state_dict(),
                       os.path.join(save_dir, 'Diffusion_Epoch%d-GLoss%.4f.pth' % (epoch + 1, total_loss)))
