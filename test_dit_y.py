import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import datetime
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from dataload import AirFoilMixParsec
import numpy as np
from models import script_utils,VAE
from utils import Fit_airfoil,vis_airfoil2,de_norm,calculate_smoothness,cal_diversity_score, norm
import os


def get_datasets():
    """获得训练、验证 数据集"""
    train_dataset = AirFoilMixParsec(split='train')
    val_dataset = AirFoilMixParsec(split='val')
    return train_dataset, val_dataset


# airfoil generatoin condition on keypoint + parsec 

def main():
    args = create_argparser().parse_args()
    device = args.device
    

    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.model_checkpoint is not None:
        diffusion.load_state_dict(torch.load(args.model_checkpoint))
    if args.optim_checkpoint is not None:
        optimizer.load_state_dict(torch.load(args.optim_checkpoint))

    batch_size = args.batch_size
    train_dataset, test_dataset = get_datasets()

    train_loader = script_utils.cycle(DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
    ))
    test_bs = 50
    test_loader = DataLoader(test_dataset, batch_size=test_bs, drop_last=False, num_workers=8, shuffle=False)
    
    print(f"Num of test set: {len(test_dataset)}")
    smooth_sum = 0
    num_sum = 0
    gen_gt = None
    
    total_parsec_loss = [[0]*3 for _ in range(11)]

        
    correct_pred = 0  # 预测正确的样本数量
    total_pred = 0  # 总共的样本数量
    total_loss = 0.0
    
    with torch.no_grad():
        diffusion.eval()
        for i,data in enumerate(tqdm(test_loader)):
            if i > 10:
                break
            y = data['params'] # (B, 11)
            y2 = de_norm(data['keypoint'],dtype='tensor')[:,:,1:2] # (B, 26, 1)
            y2 = y2.reshape(-1, 26)

            y = y.to(device) # (B, 11)
            y2 = y2.to(device) # (B, 26)
            
            samples = diffusion.sample_ddim(batch_size=y.shape[0], device=device, y=y, y2=y2).to(device).reshape(y.shape[0], 257,1)
            samples = samples.cpu().numpy()
            ref_gt = de_norm(data['gt'].cpu().numpy())
            samples = np.concatenate((ref_gt[:,:,0:1], samples),axis=-1)
            
            # gen_gt = samples.seq
            for bi in range(y.shape[0]):
                smooth = calculate_smoothness(samples[bi])
                smooth_sum += smooth
                num_sum += 1
                source = de_norm(data['gt'][bi].cpu().numpy())
                # vis_airfoil2(source,samples[bi],f'Test_{i*test_bs+bi}',dir_name='./samples_outputs/dit_xy/',sample_type='ddim')
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
                # 给他们拼接同一个x坐标
                source = samples[idx][:,1] # [257]
                target = ref_gt[idx][:,1] # [257]
                
                # 需要check 一下为啥x直接就是numpy格式
                source = np.stack([ref_gt[idx][:,0],source],axis=1)
                np.savetxt('output.txt', source, delimiter='\t', fmt='%.6f', header='Column1\tColumn2', comments='')
                breakpoint()
                target = np.stack([ref_gt[idx][:,0],target],axis=1)
                source_parsec = Fit_airfoil(source).parsec_features
                target_parsec = Fit_airfoil(target).parsec_features
                for i in range(11):
                    total_parsec_loss[i][0] += abs(source_parsec[i]-target_parsec[i]) # 绝对误差
                    total_parsec_loss[i][1] += abs(source_parsec[i]-target_parsec[i])/(abs(target_parsec[i])+1e-9) # 相对误差
                    total_parsec_loss[i][2] += abs(target_parsec[i]) # 真实值的绝对值
    
    total_pred = num_sum
    accuracy = correct_pred / total_pred
    avg_loss = total_loss / total_pred
    avg_parsec_loss = [(x/total_pred,y/total_pred,z/total_pred) for x,y,z in total_parsec_loss]
    # 将avg_parsec_loss中的每个元素转换为科学计数法，保留两位有效数字
    avg_parsec_loss_sci = [f"{x:.2e}" for x, y, z in avg_parsec_loss]
            
    diver = cal_diversity_score(gen_gt)
    print(f"Num Gen: {gen_gt.shape[0]}")
    print(f"accuracy: {accuracy}, avg_loss: {avg_loss}")
    print(f"avg_parsec_loss: {' & '.join(avg_parsec_loss_sci)}")
    print(f"Avg Smoothness: {smooth_sum/num_sum}")
    print(f"Avg diversity: {diver}")



def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=2e-5,
        batch_size=256,
        iterations=300000,
        log_to_wandb=False,
        log_rate=100000,
        checkpoint_rate=100000,
        log_dir="weights/dit_xy",
        project_name='airfoil-dit-xy',
        run_name=run_name,
        start_iter=1,
        model_checkpoint="/home/bingxing2/ailab/scxlab0058/airfoil/DiffAirfoil-main/weights/dit_y_vicinal/model-airfoil-290000.pth",
        optim_checkpoint="/home/bingxing2/ailab/scxlab0058/airfoil/DiffAirfoil-main/weights/dit_y_vicinal/optim-airfoil-290000.pth",
        schedule_low=1e-4,
        schedule_high=0.02,
        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()