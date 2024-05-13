from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataload import AirFoilDatasetParsec 
from utils import vis_airfoil, vis_airfoils, eval, vis_traj
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from models import VAE
from models.dit import PointDiT
from models.unet import Unet
from models.latent_diffusion import AirfoilDiffusion
from sklearn.decomposition import PCA
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_choices = {
    "cst_gen": ['cst_gen'],
    "supercritical_airfoil": ['supercritical_airfoil'],
    "interpolated_uiuc": ['interpolated_uiuc'],
    "afbench": ['cst_gen', 'supercritical_airfoil', 'interpolated_uiuc'] 
}

parser = argparse.ArgumentParser()

parser.add_argument('--timesteps', type=int, default=500, help='Number of timesteps')
parser.add_argument('--cond_dim', type=int, default=37)
parser.add_argument('--model_type', choices=['diff_raw', 'dit_raw', 'diff_latent', 'dit_latent'], default='diff_latent')

parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
parser.add_argument('--eval_freq', type=int, default=200, help='Evaluation frequency')
parser.add_argument('--save_freq', type=int, default=500, help='Model saving frequency')

parser.add_argument('--data_type', choices=['cst_gen', 'supercritical_airfoil', 'interpolated_uiuc', 'afbench'], default='interpolated_uiuc')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')

parser.add_argument('--root_dir', type=str, default='/home/bingxing2/ailab/scxlab0058/airfoil/DiffAirfoil-dev/new_weights', help='Root directory for saving weights')
parser.add_argument('--checkpoint_path', type=str, default=None)
parser.add_argument('--exp_name', type=str, default='', help='Directory name for logging relative to root_dir')

parser.add_argument('--test_only', default=False, action='store_true')
args = parser.parse_args()

args.log_dir = os.path.join(args.root_dir, f"{args.model_type}-{args.data_type}-{args.exp_name}")
os.makedirs(args.log_dir, exist_ok=True)
with open(f"{args.log_dir}/config.json", 'w') as f:
    json.dump(vars(args), f, indent=4)

dataset_name = data_choices[args.data_type]
train_dataset = AirFoilDatasetParsec(split='train', dataset_names=dataset_name)
val_dataset = AirFoilDatasetParsec(split='test', dataset_names=data_choices["afbench"])
train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
print(f"Num of Train: {len(train_dataset)}")
print(f"Num of Val: {len(val_dataset)}")

if "raw" in args.model_type:
    raise NotImplementedError("This feature is not implemented yet.")
vae = VAE()
checkpoint = torch.load('/home/bingxing2/ailab/scxlab0058/airfoil/DiffAirfoil-dev/new_weights/vae/ckpt_epoch_1200.pth', map_location='cpu')
vae.load_state_dict(checkpoint['model'])
vae = vae.to(device)
vae.eval()

if "dit" in args.model_type:
    model_in = PointDiT(
        latent_size = 32,
        input_channels = 1,
        hidden_size=256,
        condition_size1=args.cond_dim,
        )
    args.lr = 5e-5 # changed
else:
    model_in = Unet(timesteps=args.timesteps,time_embedding_dim=256,in_channels=1,out_channels=1,base_dim=32,dim_mults=[1, 2, 4, 8], cond_dim=args.cond_dim)
print(model_in)
model = AirfoilDiffusion(model_in, 32, 1, 1, timesteps=args.timesteps).to(device)

optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10,min_lr=0.00005)
loss_fn = nn.MSELoss()

    
if not args.test_only:
    start_epoch = 0
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(device)
        args.log_dir = args.log_dir + "cont"
        print(f"Log dir: {args.log_dir}")
        start_epoch = 1001

    raw_cond = []
    for data in train_loader:
        y1= data['params'] # (B, 11)
        y2 = data['keypoint'][:,:,1] # (B, 26)
        cond = torch.cat((y1,y2), dim=-1)
        raw_cond.append(cond)
    
    raw_cond = torch.cat(raw_cond, dim=0)
    pca = PCA(n_components=2)
    pca.fit(raw_cond)
    
    print("Start training")
    for i in tqdm(range(start_epoch, args.epochs)):
        model.train()
        epoch_loss = 0
        for data in train_loader:
            gt = data['gt'][:,:,1] # (B, 257)
            y1= data['params'] # (B, 11)
            y2 = data['keypoint'][:,:,1] # (B, 26)

            gt = gt.to(device) # (B, 257, 1)
            y1 = y1.to(device) # (128, 11)
            y2 = y2.to(device)

            mu, _ = vae.encode(gt)
            mu = mu.to(device).unsqueeze(1)

            noise = torch.randn_like(mu, device=device)
            reduced_cond = torch.tensor(pca.transform(torch.cat((y1,y2), dim=-1).cpu()),dtype=torch.float32).to(device)
            pred = model(mu,noise,y1,y2, reduced_cond)
            loss = loss_fn(pred,mu)
            if "dit" in args.model_type:
                loss = torch.clamp(loss, min=None, max=10)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)
        print(f"epoch {i} loss {epoch_loss} lr {optimizer.param_groups[0]['lr']}")

        if (i+1) % args.save_freq == 0:
            spath = os.path.join(args.log_dir, f"ckpt_epoch_{i+1}.pth")
            torch.save(model.state_dict(), spath)
        if (i+1) % args.eval_freq == 0:
            eval(model, vae, val_dataset, args.log_dir, i+1)
else:
    print("Loading checkpoint")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)

# vis_traj(model, vae, val_dataset, args.log_dir, "Vis_traj")

from utils import Fit_airfoil,calculate_smoothness,cal_diversity_score

val_loader2 = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=512,
                            num_workers=args.num_workers)

smooth_sum = 0
div_sum = 0
num_sum = 0
gen_gt = None
total_parsec_loss = [[0]*3 for _ in range(11)]
correct_pred = 0  
total_pred = 0  
total_loss = 0.0

with torch.no_grad():
    model.eval()
    for i,data in enumerate(tqdm(val_loader2)):
        gt = data['gt'][:,:,1] # (B, 257)
        y1= data['params'] # (B, 11)
        y2 = data['keypoint'][:,:,1] # (B, 26)

        gt = gt.to(device) # (B, 257, 1)
        y1 = y1.to(device) # (128, 11)
        y2 = y2.to(device)
        
        samples, all_samples = model.sampling(y1.shape[0], y1, y2, device)
        samples = samples.squeeze()
        samples = vae.decode(samples).detach().cpu().numpy()
        samples = np.expand_dims(samples, -1)

        ref_gt = data['gt'].cpu().numpy()
        samples = np.concatenate((ref_gt[:,:,0:1], samples),axis=-1)

        if gen_gt is None:
            gen_gt = samples
        else:
            gen_gt = np.concatenate((gen_gt, samples), axis=0)


        loss = np.mean((ref_gt[:,::10] - samples[:,::10]) ** 2, axis=(1,2))
        total_loss += np.sum(loss)
        # 判断样本是否预测正确
        distances = torch.norm(torch.tensor(ref_gt - samples),dim=-1) #(B,257)
    
        # 点的直线距离小于t，说明预测值和真实值比较接近，认为该预测值预测正确
        t = 0.01
        # 257个点中，预测正确的点的比例超过ratio，认为该形状预测正确
        ratio = 0.75
        count = (distances < t).sum(dim=1) #(B) 一个样本中预测坐标和真实坐标距离小于t的点的个数
        correct_count = (count >= ratio*257).sum().item() # batch_size数量的样本中，正确预测样本的个数
        correct_pred += correct_count

        # 统计一下物理量之间的误差
        for idx in range(samples.shape[0]):
            smooth = calculate_smoothness(samples[idx])
            smooth_sum += smooth
            num_sum += 1
            # 给他们拼接同一个x坐标
            source = samples[idx][:,1] # [257]
            target = ref_gt[idx][:,1] # [257]
            
            # 需要check 一下为啥x直接就是numpy格式
            source = np.stack([ref_gt[idx][:,0],source],axis=1)
            target = np.stack([ref_gt[idx][:,0],target],axis=1)
            source_parsec = Fit_airfoil(source).parsec_features
            target_parsec = Fit_airfoil(target).parsec_features
            for i in range(11):
                total_parsec_loss[i][0] += abs(source_parsec[i]-target_parsec[i]) # 绝对误差
                total_parsec_loss[i][1] += abs(source_parsec[i]-target_parsec[i])/(abs(target_parsec[i])+1e-9) # 相对误差
                total_parsec_loss[i][2] += abs(target_parsec[i]) # 真实值的绝对值

        # for idx in tqdm(range(samples.shape[0]),desc="Cal diver score"):
        #     y1_same = y1[idx].unsqueeze(0).repeat(100,1).to(device)
        #     y2_same = y2[idx].unsqueeze(0).repeat(100,1).to(device)
            
        #     samples_same, _ = model.sampling(y1_same.shape[0], y1_same, y2_same, device)
        #     samples_same = samples_same.squeeze()
        #     samples_same = vae.decode(samples_same).detach().cpu().numpy()
        #     samples_same = np.expand_dims(samples_same, -1)

        #     ref_gt = data['gt'][idx].unsqueeze(0).repeat(100,1,1).cpu().numpy()
        #     samples_same = np.concatenate((ref_gt[:,:,0:1], samples_same),axis=-1)
        #     div_sum += cal_diversity_score(samples_same)

total_pred = num_sum
accuracy = correct_pred / total_pred
avg_loss = total_loss / total_pred
avg_parsec_loss = [(x/total_pred,y/total_pred,z/total_pred) for x,y,z in total_parsec_loss]
avg_parsec_loss_sci = [f"{x:.2e}" for x, y, z in avg_parsec_loss]

print(f"Num Gen: {gen_gt.shape[0]}")
print(f"accuracy: {accuracy}, avg_loss: {avg_loss}")
print(f"avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}")
print(f"Avg Smoothness: {smooth_sum/num_sum}")
print(f"Avg diversity: {div_sum/num_sum}")

spath = os.path.join(args.log_dir, f"eval_res.txt")
with open(spath, "w") as file:
    file.write(f"Num Gen: {gen_gt.shape[0]}")
    file.write(f"accuracy: {accuracy}, avg_loss: {avg_loss}")
    file.write(f"avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}")
    file.write(f"Avg Smoothness: {smooth_sum/num_sum}")
    file.write(f"Avg diversity: {div_sum/num_sum}")
