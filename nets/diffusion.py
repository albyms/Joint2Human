import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy
from omegaconf import OmegaConf

from utils.visualize import visualize_inter_res
from nets.autoencoderKL import AutoencoderKL

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)

class GaussianDiffusion(nn.Module):
    def __init__(
        self, model, img_size, img_channels, num_classes=None, betas=[], loss_type="l2", ema_decay=0.9999, ema_start=2000, ema_update_rate=1,
    ):
        super().__init__()
        self.model      = model
        self.ema_model  = deepcopy(model)

        self.ema                = EMA(ema_decay)
        self.ema_decay          = ema_decay
        self.ema_start          = ema_start
        self.ema_update_rate    = ema_update_rate
        self.step               = 0

        self.img_size       = img_size
        self.img_channels   = img_channels
        self.num_classes    = num_classes

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type      = loss_type
        self.num_timesteps  = len(betas)

        alphas              = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # betas             [0.0001, 0.00011992, 0.00013984 ... , 0.02]
        self.register_buffer("betas", to_torch(betas))
        # alphas            [0.9999, 0.99988008, 0.99986016 ... , 0.98]
        self.register_buffer("alphas", to_torch(alphas))
        # alphas_cumprod    [9.99900000e-01, 9.99780092e-01, 9.99640283e-01 ... , 4.03582977e-05]
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        # sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        # sqrt(1 - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        # sqrt(1 / alphas)
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))
    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)

    @torch.no_grad()
    def ddim_remove_noise(self,x,t,y,w,use_ema,eta=0.0):
        if not use_ema:
            pred_eps = self.model(x,t,y)
        else:
            pred_eps = self.ema_model(x,t,y)
        
        x_start = (x - pred_eps * extract(self.sqrt_one_minus_alphas_cumprod,t,x.shape))/extract(self.sqrt_alphas_cumprod,t,x.shape) 
        x_start = x_start.clamp(-1, 1)
        
        alpha_bar_prev = extract(self.alphas_cumprod,t-1,x.shape) if t > 0 else torch.tensor([1.0]).to('cuda:0')
        alpha_bar = extract(self.alphas_cumprod,t,x.shape)

        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            x_start * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * pred_eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    @torch.no_grad()
    def remove_noise_guidance(self, x, t, y,w,use_ema):
        cond_shape = y.shape
        tmp_y = torch.zeros(cond_shape, device = y.device)
        if not use_ema:
            pred_eps_cond = self.model(x,t,y)
            pred_eps_uncond = self.model(x,t,tmp_y)
        else:
            pred_eps_cond = self.ema_model(x,t,y)
            pred_eps_uncond = self.ema_model(x,t,tmp_y)
        pred_eps = (1+w) * pred_eps_cond - w * pred_eps_uncond
        return (
            (x - extract(self.remove_noise_coeff, t, x.shape) * pred_eps *
            extract(self.reciprocal_sqrt_alphas, t, x.shape)
        ))


    @torch.no_grad()
    def remove_noise(self, x, t, joint,iuv, use_ema=True):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, joint,iuv)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, joint,iuv)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
    @torch.no_grad()
    def sample(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x.cpu().detach()
    
    @torch.no_grad()
    def sample_vis(self, batch_size, device, y=None, use_ema=True):

        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        # define pre_model
        config_pos = "xxx" 
        config = OmegaConf.load(config_pos)
        weight_path = "xxx"

        # load pre_model
        pre_model = AutoencoderKL(**config.model.get("params", dict()))
        pre_model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        pre_model.to(device)
        pre_model = pre_model.eval()
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        index = 1
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            if (t+1) % 10 == 0 or t == 0:   
                with torch.no_grad():
                    de_res_ceof  = pre_model.decode(z=x)
                    visualize_inter_res(de_res_ceof,device,index)
                    index += 1
        return de_res_ceof

    @torch.no_grad()
    def resample(self,x,t_noise,t_denoise,device, y=None, use_ema=True):
        noise = torch.randn_like(x)
        t = torch.tensor([t_noise],dtype=int,device=device)

        perturbed_x  = self.perturb_x(x, t, noise)
        x = perturbed_x

        for t in range(t_denoise - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(x.shape[0])
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x.cpu().detach()

    @torch.no_grad() 
    def sample_test(self, batch_size, device,steps=None,joint=None,iuv=None,use_ema=True):
        # if y is not None and batch_size != len(y):
        #     raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        for t in range(steps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, joint,iuv, use_ema)
            
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x.cpu().detach()
    
    @torch.no_grad() 
    def ddim_sample(self, batch_size,device,ddim_timesteps,y=None, use_ema=True,eta=0.0):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)

        for t in reversed(ddim_timesteps):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.ddim_remove_noise(x, t_batch, y, use_ema,eta)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x.cpu().detach()

    @torch.no_grad() 
    def sample_guidance(self, batch_size, device,steps=None,y=None,w=4.0,use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        for t in range(steps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            print(f'w = {w}')
            x = self.remove_noise_guidance(x, t_batch,y,w,use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x.cpu().detach()

    @torch.no_grad()
    def sample_mix_test(self, batch_size,device,base=None,mix_rate=None,steps=None,y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        base = base.to(device)
    
        if base != None:
            x = x + mix_rate * base
        for t in range(steps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x.cpu().detach()
    
    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence
    def perturb_x(self, x, t, noise):
        alpha = extract(self.sqrt_alphas_cumprod, t,  x.shape)
        beta = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return (
            alpha * x + beta * noise
        )   

    def get_losses(self, x, t, joint=None,iuv=None):
        noise           = torch.randn_like(x)
        perturbed_x     = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, joint=joint,iuv=iuv)

        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)
        return loss

    def forward(self, x, joint=None,iuv=None):
        b, c, h, w  = x.shape
        device      = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, joint=joint,iuv=iuv)

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)