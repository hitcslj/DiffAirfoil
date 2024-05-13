import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler
from dataload import AirFoilDatasetParsec 
import math 
import numpy as np
from utils import vis_airfoil, vis_airfoils, visualize_latent_space
from models import VAE


def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--feature_size', type=int, default=257)
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch Size during training')
    parser.add_argument('--latent_size', type=int, default=32,
                        help='Batch Size during training')
    parser.add_argument('--num_workers',type=int,default=8)
    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                          choices=["step", "cosine"])

    # io
    parser.add_argument('--checkpoint_path', default='',help='Model checkpoint path') # ./eval_result/logs_p/ckpt_epoch_last.pth
    parser.add_argument('--log_dir', default='new_weights/vae',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--val_freq', type=int, default=200)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=200)  # epoch-wise
    

    # 评测指标相关
    parser.add_argument('--distance_threshold', type=float, default=0.01) # xy点的距离小于该值，被认为是预测正确的点
    parser.add_argument('--threshold_ratio', type=float, default=0.75) # 200个点中，预测正确的点超过一定的比例，被认为是预测正确的样本
   
    args, _ = parser.parse_known_args()

    return args

# BRIEF load checkpoint.
def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint."""
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    try:
        args.start_epoch = int(checkpoint['epoch'])
    except Exception:
        args.start_epoch = 1
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint['epoch']
    ))

    del checkpoint
    torch.cuda.empty_cache()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar,beta=1.0):
    recons = nn.MSELoss(reduction='sum')(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recons + beta * KLD

# BRIEF save model.
def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        os.makedirs(args.log_dir, exist_ok=True)
        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


class Trainer:
    
    def get_datasets(self):
        """获得训练、验证 数据集"""
        train_dataset = AirFoilDatasetParsec(split='train', dataset_names=['cst_gen', 'supercritical_airfoil', 'interpolated_uiuc'])
        val_dataset = AirFoilDatasetParsec(split='test', dataset_names=['cst_gen', 'supercritical_airfoil', 'interpolated_uiuc'])
        self.val_dataset = val_dataset
        print(f"Num of Train: {len(train_dataset)}")
        print(f"Num of Val: {len(val_dataset)}")
        return train_dataset, val_dataset
    
    def get_loaders(self,args):
        """获得训练、验证 dataloader"""
        print("get_loaders func begin, loading......")
        train_dataset, val_dataset = self.get_datasets()
        train_loader = DataLoader(train_dataset,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset,
                                  shuffle=False,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        self.val_loader = val_loader
        return train_loader,val_loader

    @staticmethod
    def get_model(args):
        model = VAE(args.feature_size, args.latent_size)
        return model

    @staticmethod
    def get_optimizer(args, model):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return optimizer
    


    def train_one_epoch(self,model,optimizer,dataloader,device,epoch,beta):
        """训练一个epoch"""
        model.train()  # set model to training mode
        for _,data in enumerate(tqdm(dataloader)):
            gt = data['gt'][:, :, 1] # [b,257,1]
            gt = gt.to(device)
            # # AE
            recon_batch, mu, logvar = model(gt) # [b,257,1]
            recon_batch = recon_batch.reshape(gt.shape[0], -1)
            loss = loss_function(recon_batch, gt, mu, logvar, beta=beta)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 参数更新
            optimizer.step()            
        # 打印loss
        print('====> Epoch: {} Average loss: {:.8f}'.format(
          epoch, loss.item()))


    @torch.no_grad()
    def evaluate_one_epoch(self, model, dataloader,device, epoch, args):
        """验证一个epoch"""
        model.eval()
        
        correct_pred = 0  # 预测正确的样本数量
        total_pred = 0  # 总共的样本数量
        total_loss = 0.0

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
            gt = data['gt'][:,:,1] # [b,257]
            gt = gt.to(device)
            # AE
            recon_batch, mu, logvar = model(gt) # [b,257]
            recon_batch = recon_batch.reshape(gt.shape[0], -1)
            total_pred += gt.shape[0]
            loss = loss_function(recon_batch, gt, mu, logvar)
            total_loss += loss.item()
            distances = gt.cpu() - recon_batch.cpu() #(B,257)
            # 点的直线距离小于t，说明预测值和真实值比较接近，认为该预测值预测正确
            t = args.distance_threshold
            # 200个点中，预测正确的点的比例超过ratio，认为该形状预测正确
            ratio = args.threshold_ratio
            count = (distances < t).sum(dim=1) #(B) 一个样本中预测坐标和真实坐标距离小于t的点的个数
            correct_count = (count >= ratio*257).sum().item() # batch_size数量的样本中，正确预测样本的个数
            correct_pred += correct_count
            
        accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        
        print(f"eval——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}")

    @torch.no_grad()
    def infer(self, model,device,epoch,args):
        model.eval()
        sample_num = 32
        real_airfoils = [self.val_dataset.__getitem__(i)['gt'][:,1] for i in range(sample_num)]
        real_airfoils_tensor = torch.stack(real_airfoils).to(device)
        recon_airfoils,_,_ = model(real_airfoils_tensor)
        recon_airfoils = recon_airfoils.reshape(sample_num, -1).cpu().numpy()

        noise = torch.randn((sample_num,args.latent_size)).to(device) 
        airfoil = model.decode(noise).reshape(sample_num,args.feature_size)
        airfoil = airfoil.cpu().numpy()
        gt_x = self.val_dataset.__getitem__(0)['gt'][:,0:1].cpu().numpy()
        os.makedirs('logs/cvae',exist_ok=True)
        vis_airfoils(gt_x, np.expand_dims(real_airfoils_tensor.cpu().numpy(), -1),f"{epoch}_Real",dir_name=args.log_dir,title="Real Airfoils")
        vis_airfoils(gt_x, np.expand_dims(recon_airfoils, -1),f"{epoch}_Reconstructed",dir_name=args.log_dir,title="Reconstructed Airfoils")
        vis_airfoils(gt_x, np.expand_dims(airfoil, -1),f"{epoch}_Synthesized",dir_name=args.log_dir,title="Synthesized Airfoils")
        visualize_latent_space(model, self.val_loader, device, f"{epoch}_Latent",dir_name=args.log_dir)

    def main(self,args):
        """Run main training/evaluation pipeline."""
        # 单卡训练
        model = self.get_model(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        train_loader, val_loader = self.get_loaders(args) 
        optimizer = self.get_optimizer(args,model)
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / args.max_epoch)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
      
        # Check for a checkpoint
        if len(args.checkpoint_path)>0:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
        beta_start = 0.0
        beta_end = 0.0005
        beta_increment = (beta_end - beta_start) / args.max_epoch
        for epoch in tqdm(range(args.start_epoch,args.max_epoch+1)):
            current_beta = beta_start + beta_increment * epoch
            # train
            self.train_one_epoch(model=model,
                                 optimizer=optimizer,
                                 dataloader=train_loader,
                                 device=device,
                                 epoch=epoch,
                                 beta=current_beta
                                 )
            scheduler.step()
            # save model and validate
            # args.val_freq = 1
            if epoch % args.val_freq == 0:
                save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Validation begin.......")
                self.evaluate_one_epoch(
                    model=model,
                    dataloader=val_loader,
                    device=device, 
                    epoch=epoch, 
                    args=args)
                self.infer(model=model,
                           device=device,
                           epoch=epoch, 
                           args=args)

if __name__ == '__main__':
    opt = parse_option()
    torch.backends.cudnn.enabled = True
    # 启用cudnn的自动优化模式
    torch.backends.cudnn.benchmark = True
    # 设置cudnn的确定性模式，pytorch确保每次运行结果的输出都保持一致
    torch.backends.cudnn.deterministic = True

    trainer = Trainer()
    trainer.main(opt)