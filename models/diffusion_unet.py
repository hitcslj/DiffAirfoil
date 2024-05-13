import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

from functools import partial
from copy import deepcopy

from .ema import EMA
from .utils import extract
from tqdm import tqdm


class PointUnetDiffusion(nn.Module):
    __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    Args:
        model (nn.Module): model which estimates diffusion noise
        img_size (tuple): image size tuple (H, W)
        img_channels (int): number of image channels
        betas (np.ndarray): numpy array of diffusion betas
        loss_type (string): loss type, "l1" or "l2"
        ema_decay (float): model weights exponential moving average decay
        ema_start (int): number of steps before EMA
        ema_update_rate (int): number of steps before each EMA update
    """
    def __init__(
        self,
        model,
        latent_size,
        channels,
        betas,
        loss_type="l2",
        ema_decay=0.9999,
        ema_start=5000,
        ema_update_rate=1,
    ):
        super().__init__()
        self.self_condition = False
        self.model = model
        self.objective = "pred_noise"
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0

        self.latent_size = latent_size
        self.channels = channels
 

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")

        self.loss_type = loss_type
        self.num_timesteps = len(betas)
        self.ddim_timesteps = 50
        self.ddim_eta = 0

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self.y_embedder = nn.Linear(11, self.latent_size)
        self.y2_embedder = nn.Linear(26, self.latent_size)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(1. / alphas_cumprod - 1))
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
    def remove_noise(self, x, t, y, y2, use_ema=True):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y, y2)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y, y2)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        
    @torch.no_grad()
    def sample_ddim(self, batch_size, device, y=None, y2=None, use_ema=True,clip_denoised=True, sample_steps=None):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        if sample_steps is None:
          ddim_timesteps = self.ddim_timesteps
        else:
          ddim_timesteps = sample_steps
        c = self.num_timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.num_timesteps, c)))

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        # start from pure noise (for each example in the batch)
        x = torch.randn(batch_size, self.latent_size, self.channels, device=device)
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, x.shape)
    
            # 2. predict noise using model
            if use_ema:
                pred_noise = self.ema_model(x, t, y, y2)
            else:
                pred_noise = self.model(x,t,condition, x_self_cond=x_self_cond)
            
            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)

            x = x_prev
            
        return x.cpu().detach()


    @torch.no_grad()
    def sample_ddim_sequence(self, batch_size, device, y=None, y2=None, use_ema=True,clip_denoised=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        ddim_timesteps = self.ddim_timesteps
        c = self.num_timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.num_timesteps, c)))

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        # start from pure noise (for each example in the batch)
       
        x = torch.randn(batch_size, self.latent_size, self.channels, device=device)
        ans = [x]
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, t, x.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, x.shape)
    
            # 2. predict noise using model
            if use_ema:
                pred_noise = self.ema_model(x, t, y, y2)
            else:
                pred_noise = self.model(x,t,y,y2)
            
            # 3. get the predicted x_0
            pred_x0 = (x - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)

            x = x_prev
            ans.append(x)
            
        return ans
    
    @torch.no_grad()
    def sample(self, batch_size, device, y=None,y2=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size,self.latent_size,self.channels, device=device)
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, y2,use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.latent_size, self.channels, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        return diffusion_sequence

    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )   


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def q_sample(self, x_start, t, noise=None):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, time_steps, condition):
        if x_start is None or time_steps is None:
            raise ValueError("x_start and time_steps must be provided")
        
        b, c, n = x_start.shape
        noise = torch.randn_like(x_start)

        # Use time_steps in sampling or other operations
        x = self.q_sample(x_start=x_start, t=time_steps, noise=noise) #noise_x

        # Condition on cl_cd if needed
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, time_steps, condition).pred_x_start
                x_self_cond.detach_()
        else:
            x_self_cond = None

        # Forward through the model (assuming model accepts condition if necessary)
        model_out = self.model(x, time_steps, condition, x_self_cond=x_self_cond)

        # Determine the target based on the diffusion objective
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # Adjust loss weight if needed based on time_steps
        loss = loss * extract(self.loss_weight, time_steps, loss.shape)
        return loss.mean()

    def forward(self, x, y=None,y2=None):
        b, c, n = x.shape
        device = x.device

        t = torch.randint(0, self.num_timesteps, (b,), device=device) # (128)
        condition = self.y_embedder(y) + self.y2_embedder(y2)# torch.cat((self.y_embedder(y).unsqueeze(1), self.y2_embedder(y2).unsqueeze(1)), dim = 1)
        return self.p_losses(x, t, condition.unsqueeze(1))
        # return self.get_losses(x, t, y, y2)