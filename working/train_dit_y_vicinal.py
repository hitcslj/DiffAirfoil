import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from scipy.spatial import KDTree

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
            num_workers=8,
            ))
            
        #Build KD-Tree
        all_data = [torch.cat((train_dataset[i]['params'], de_norm(train_dataset[i]['keypoint'],dtype='tensor')[:,1]),dim=-1) for i in range(len(train_dataset))]
        all_y = torch.stack(all_data)
        all_data_np = all_y.numpy()
        all_x = torch.stack([de_norm(train_dataset[i]['gt'],dtype='tensor')[:,1:2] for i in range(len(train_dataset))])
        kd_tree = KDTree(data=all_data_np)
        
        test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False, num_workers=8, shuffle=False)

        for iteration in tqdm(range(args.start_iter, args.iterations + 1)):
            diffusion.train()
            y = np.random.choice(np.array(range(all_data_np.shape[0])), size=batch_size, replace=False)
            y = all_data_np[y]
            # hyper param: threshold, v, sigma(noise)
            loss = diffusion.vicinal_train(train_dataset, kd_tree, y, all_x, all_y, threshold=10,sigma=0.1, device=device)

            # loss = diffusion(gt, y, y2)
            acc_train_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()
            print(f"iteration: {iteration}, train_loss: {acc_train_loss}")

            # Debug:
            # args.log_rate = 1
            if iteration % args.log_rate == 0:
                with torch.no_grad():
                    diffusion.eval()
                    for i,data in enumerate(tqdm(test_loader)):
                        if i > 10: 
                          break
 
                        gt = data['gt'] # (B, 257, 2)
                        y = data['params'] # (B, 11)
                        y2 = de_norm(data['keypoint'],dtype='tensor')[:,:,1:2] # (B, 26, 1)
                        y2 = y2.reshape(-1, 26)

                        y = y.to(device) # (128, 11)
                        y2 = y2.to(device) # (128, 26)

                        samples = diffusion.sample_ddim(batch_size=1, device=device, y=y, y2=y2).to(device).reshape(257,1)
                        samples = samples.cpu().numpy()
            
                        source = de_norm(data['gt'][0].cpu().numpy())
                        samples = np.concatenate((source[:,0:1], samples),axis=-1)
                        
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
            learning_rate=1e-3,
            batch_size=256,
            iterations=600000,
            log_to_wandb=False,
            log_rate=1000,
            checkpoint_rate=10000,
            log_dir="weights/dit_y_vicinal",
            project_name='airfoil-dit-y-vicinal',
            run_name=run_name,
            start_iter=300000,
            model_checkpoint="/home/bingxing2/ailab/scxlab0058/airfoil/DiffAirfoil-main/weights/dit_y_vicinal/model-airfoil-300000.pth",
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
