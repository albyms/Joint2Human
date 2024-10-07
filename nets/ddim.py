import torch
from torch import nn
import numpy as np
from PIL import Image

from nets import (GaussianDiffusion, UNet, generate_cosine_schedule,
                  generate_linear_schedule)

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


class DDIMSampler(object):

    def __init__(self,model_path,device,ddim_num_steps=250,schedule="linear",**kwargs):
        super().__init__()
        self.model_path= model_path
        self.channel=128
        self.input_shape= (128, 128)
        self.schedule=schedule

        self.ddpm_model = GaussianDiffusion(UNet(8, self.channel), self.input_shape, 8,betas=np.linspace(2e-4, 0.04, 1000))
        self.ddpm_model_init(device=device)
        self.ddpm_num_timesteps = self.ddpm_model.num_timesteps

        self.make_schedule(ddim_num_steps=ddim_num_steps)
        
    def ddpm_model_init(self,device):

        # 删掉不匹配的权重
        model_dict      = self.ddpm_model.state_dict()
        pretrained_dict = torch.load(self.model_path, map_location="cpu")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)

        self.ddpm_model.load_state_dict(model_dict)
        self.ddpm_model = self.ddpm_model.eval()
        print('{} model loaded.'.format(self.model_path))
        self.ddpm_model.to(device)   


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)

    def ddim_generate_ceof(self,device,y=None):
        with torch.no_grad():
            randn_in = torch.randn((1, 1)).to(device)
            test_ceof = self.ddpm_model.ddim_sample(1, randn_in.device,ddim_timesteps=self.ddim_timesteps,y=y,use_ema=False)
        return test_ceof 