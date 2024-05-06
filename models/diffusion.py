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


class PointDiTDiffusion(nn.Module):
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

        self.model = model
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

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
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

    def get_losses(self, x, t, y, y2):
        noise = torch.randn_like(x)

        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, y, y2)
        # Unet
        # estimated_noise = self.model(perturbed_x, t)
        
        # x_0_pred = (x - torch.sqrt((1. - extract(self.alphas_cumprod, t, x.shape))) * estimated_noise) / torch.sqrt(extract(self.alphas_cumprod, t, x.shape))

        # cal x_0_pred physical loss
      

        # loss_physical = 0
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)

        # return loss + loss_physical
        return loss
    

    def vicinal_loss(self, x, t, y, y2, strategy = "soft", sigma=1, kappa=0.1, nonzero_soft_weight_threshold=0.1):
        batch_size = x.shape[0]
        # 1. randomlt sample u^i from C
        random_index = torch.randint(0, batch_size, (batch_size,), device=x.device)
        cond = torch.cat((y,y2),dim=-1)
        batch_cond = cond[random_index]
        # 2. Add Guass noise to u^i 
        batch_epsilons = torch.randn(batch_size,cond.shape[-1], device=x.device) * sigma
        # 2. Get uhat^i
        batch_target_cond = torch.clamp(batch_cond + batch_epsilons, 0.0, 1.0)
        batch_real_indx = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        # batch_fake_cond = torch.zeros(batch_size)
        # 3. Find k nearest neighbors 
        for j in range(batch_size):
            if strategy == 'hard':
                indx_real_in_vicinity = torch.where(torch.abs(cond - batch_target_cond[j]) <= kappa)[0]
            else:
                threshold = -torch.log(torch.tensor(nonzero_soft_weight_threshold, device=x.device)) / kappa
                indx_real_in_vicinity = torch.where(((cond - batch_target_cond[j]) ** 2).sum(dim=1) <= threshold)[0]

            while indx_real_in_vicinity.shape[0] < 1:
                batch_target_cond[j] = torch.clamp(batch_cond[j] + torch.randn(1) * sigma, 0.0, 1.0)

                if strategy == 'hard':
                    indx_real_in_vicinity = torch.where(torch.abs(cond - batch_target_cond[j]) <= kappa)[0]
                else:
                    threshold = -torch.log(torch.tensor(nonzero_soft_weight_threshold, device=x.device)) / kappa
                    indx_real_in_vicinity = torch.where(((cond - batch_target_cond[j]) ** 2).sum(dim=1) <= threshold)[0]
            # 3. and randomly sample one (index)
            batch_real_indx[j] = indx_real_in_vicinity[torch.randint(0, indx_real_in_vicinity.size(0), (1,))]
            '''
            if self.strategy == 'hard':
                lb = batch_target_labels[j] - kappa
                ub = batch_target_labels[j] + kappa
            else:
                lb = batch_target_labels[j] - torch.sqrt(-torch.log(torch.tensor(self.nonzero_soft_weight_threshold)) / kappa)
                ub = batch_target_labels[j] + torch.sqrt(-torch.log(torch.tensor(self.nonzero_soft_weight_threshold)) / kappa)

            lb = torch.max(torch.tensor(0.0), lb)
            ub = torch.min(ub, torch.tensor(1.0))

            batch_fake_labels[j] = torch.distributions.Uniform(lb, ub).sample()
            '''
        # D^j_i
        x_real = x[batch_real_indx] 
        cond_real = cond[batch_real_indx]
        # batch_fake_labels = batch_fake_labels.unsqueeze(-1)
        # batch_target_cond = batch_target_cond.unsqueeze(-1)


        if strategy == "soft":
            weights = torch.exp(-kappa * ((batch_target_cond - cond_real) ** 2).sum(dim=1)).squeeze()
        else:
            weights = torch.ones(batch_size)
        
        noise = torch.randn_like(x_real)
        perturbed_x = self.perturb_x(x_real, t, noise)
        estimated_noise = self.model(perturbed_x, t, batch_target_cond[:,:11],batch_target_cond[:,11:])

        # if self.loss_type == "l1":
        #     loss = F.l1_loss(estimated_noise, noise)
        # elif self.loss_type == "l2":
        #     loss = F.mse_loss(estimated_noise, noise)
        loss = ((estimated_noise - noise) ** 2).mean(dim=(1,2))
        # return loss + loss_physical
        return (weights*loss).mean()
    
    def cal_W(self, y1, y2, v=0.1, strategy="soft"):
        if strategy == "soft":
            weights = torch.exp(-v * ((y1 - y2) ** 2).sum(dim=-1))
        else:
            weights = torch.ones(y1.shape[0])
        return weights

    def vicinal_train(self, train_dataset, kd_tree, y, all_x, all_y, threshold, sigma, device="cuda:0"):
        t = torch.randint(0, self.num_timesteps, (y.shape[0],), device=device)
        epsilons = np.random.normal(size=y.shape)*sigma
        y_hat = y + epsilons
        # D_i_index = kd_tree.query_ball_point(y_hat, r=threshold, workers=8) 
        _, D_i_index = kd_tree.query(y_hat, k=threshold, workers=8)
        D_i_j_index = np.array([np.random.choice(row) for row in D_i_index])
        x_i_j = all_x[D_i_j_index].to(device)
        y_i_j = all_y[D_i_j_index].to(device)
        y_hat = torch.from_numpy(y_hat).float().to(device)
        W = self.cal_W(y_hat, y_i_j, v=0.1)

        noise = torch.randn_like(x_i_j)
        perturbed_x = self.perturb_x(x_i_j, t, noise)
        estimated_noise = self.model(perturbed_x, t, y_hat[:,:11],y_hat[:,11:])
        loss = ((estimated_noise - noise) ** 2).mean(dim=(1,2))
        return (W*loss).mean()


    def forward(self, x, y=None,y2=None):
        b, N, c = x.shape
        device = x.device

        t = torch.randint(0, self.num_timesteps, (b,), device=device) # (128)
        return self.get_losses(x, t, y, y2)