import argparse
import datetime
import torch
import wandb

from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from dataload import AirFoilMixParsec
import numpy as np
from models import script_utils,VAE
from utils import Fit_airfoil,vis_airfoil2,de_norm
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
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
        os.makedirs(args.log_dir, exist_ok=True)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            run = wandb.init(
                project=args.project_name,
                entity='treaptofun',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size
        train_dataset, test_dataset = get_datasets()

        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=2,
        ))
        test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, num_workers=1, shuffle=False)
        
        for iteration in tqdm(range(args.start_iter, args.iterations + 1)):
            diffusion.train()

            data = next(train_loader)  
           
            gt = data['gt'][:,:,1:2] # (128, 257, 2)
            y = data['params'] # (128, 11)
            y2 = data['keypoint'][:,:,1] # (128, 26)
            gt = gt.to(device)  
            y = y.to(device) 
            y2 = y2.to(device)
 
            loss = diffusion(gt, y, y2)
            acc_train_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()
            print(f"iteration: {iteration}, train_loss: {acc_train_loss}")
            args.log_rate = 1
            if iteration % args.log_rate == 0:
              with torch.no_grad():
                diffusion.eval()
                for i,data in enumerate(tqdm(test_loader)):
                  x = data['gt'][0,:,0] # (257,)
                  gt = data['gt'][:,:,1:2] # (128, 257, 2)
                  y = data['params'] # (128, 11)
                  y2 = data['keypoint'][:,:,1] # (128, 26)
                  gt = gt.to(device)  
                  y = y.to(device) 
                  y2 = y2.to(device)
                  samples = diffusion.sample_ddim(batch_size=1, device=device, y=y, y2=y2).to(device)
                  samples = samples[0,:,0].cpu().numpy()
                  samples = np.stack([x,samples],axis=1)
                  samples = de_norm(samples)
                  # samples = de_norm(samples.reshape(1,257,2))
                  # samples = samples.permute(0, 2, 1).cpu().numpy()
                  # 对samples 进行可视化
                  source = de_norm(data['gt'][0].cpu().numpy())

                  vis_airfoil2(source,samples,f'{iteration}_{i}',dir_name=args.log_dir,sample_type='ddim')

            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/model-airfoil-{iteration}.pth"
                optim_filename = f"{args.log_dir}/optim-airfoil-{iteration}.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
            
        
        if args.log_to_wandb:
            run.finish()
    except KeyboardInterrupt:
        if args.log_to_wandb:
            run.finish()
        print("Keyboard interrupt, run finished early")


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
        log_dir="weights/dit",
        project_name='airfoil-dit',
        run_name=run_name,
        start_iter=1,
        model_checkpoint=None,
        optim_checkpoint=None,
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