import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
import torch.optim as optim 
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from dataload import AirFoilMixParsec 
import math 
import numpy as np
from utils import vis_airfoil,de_norm,vis_airfoil2
from models import VQ_VAE



def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size during training')
    parser.add_argument('--latent_size', type=int, default=128,
                        help='Batch Size during training')
    parser.add_argument('--num_workers',type=int,default=4)
    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                          choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[50, 75],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)

    # io
    parser.add_argument('--checkpoint_path', default='weights/vqvae_mix/ckpt_epoch_10000.pth',help='Model checkpoint path') # 
    parser.add_argument('--log_dir', default='weights/vqvae_mix',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--val_freq', type=int, default=1000)  # epoch-wise
    parser.add_argument('--save_freq', type=int, default=10000)  # epoch-wise
    

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
        train_dataset = AirFoilMixParsec(split='train')
        val_dataset = AirFoilMixParsec(split='val')
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
        return train_loader,val_loader

    @staticmethod
    def get_model(args):
        model = VQ_VAE(latent_size=args.latent_size)
        return model

    @staticmethod
    def get_optimizer(args,model):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params,
                              lr = args.lr,
                              weight_decay=args.weight_decay)
        return optimizer
    


    def train_one_epoch(self,model,optimizer,dataloader,device,epoch):
        """训练一个epoch"""
        model.train()  # set model to training mode
        for _,data in enumerate(tqdm(dataloader)):
            gt = data['gt'] # [b,257,2]
            gt = gt.to(device)
            optimizer.zero_grad()

            # # VQ-VAE
            recon_batch, z, z_quantized = model(gt) # [b,257,2],[b,37,2]

            # Reconstruction loss
            loss_recons = F.mse_loss(recon_batch, gt.view(-1, 257*2))
            # Vector quantization objective
            loss_vq = F.mse_loss(z_quantized, z.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z, z_quantized.detach())

            loss = loss_recons + loss_vq +  loss_commit
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()            
        # 打印loss, loss_recons, loss_vq, loss_commit
        print(f"train——epoch: {epoch}, loss: {loss}, loss_recons: {loss_recons}, loss_vq: {loss_vq}, loss_commit: {loss_commit}")


    @torch.no_grad()
    def evaluate_one_epoch(self, model, dataloader,device, epoch, args):
        """验证一个epoch"""
        model.eval()
        
        correct_pred = 0  # 预测正确的样本数量
        total_pred = 0  # 总共的样本数量
        total_loss = 0.0

        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader):
            gt = data['gt'] # [b,257,2]
            gt = gt.to(device)
            # # AE
            recon_batch, z, z_quantized = model(gt) # [b,257,2],[b,37,2]

            total_pred += gt.shape[0]
            loss = F.mse_loss(recon_batch, gt.view(-1, 257*2))
            
            total_loss += loss.item()
            distances = torch.norm(de_norm(gt.cpu()) - de_norm(recon_batch.reshape(-1,257,2).cpu()),dim=2) #(B,257)
            # 点的直线距离小于t，说明预测值和真实值比较接近，认为该预测值预测正确
            t = args.distance_threshold
            # 200个点中，预测正确的点的比例超过ratio，认为该形状预测正确
            ratio = args.threshold_ratio
            count = (distances < t).sum(1) #(B) 一个样本中预测坐标和真实坐标距离小于t的点的个数
            correct_count = (count >= ratio*200).sum().item() # batch_size数量的样本中，正确预测样本的个数
            correct_pred += correct_count
            
        accuracy = correct_pred / total_pred
        avg_loss = total_loss / total_pred
        
        print(f"eval——epoch: {epoch}, accuracy: {accuracy}, avg_loss: {avg_loss}")

    @torch.no_grad()
    def infer(self, model, dataloader,device, epoch, args):
        model.eval()
        test_loader = tqdm(dataloader)
        for _,data in enumerate(test_loader): # 只需要可视化张图片
            gt = data['gt'] # [b,257,2]
            gt = gt.to(device)
            recon_batch, z, z_quantized = model(gt) # [b,257,2],[b,37,2]
            for j in range(recon_batch.shape[0]):
              vis_airfoil2(de_norm(gt.cpu().numpy())[j],de_norm(recon_batch.reshape(-1,257,2).cpu().numpy())[j],epoch+j,dir_name='logs/vqvae',sample_type='vqvae_mix')
            return 

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
            
        for epoch in range(args.start_epoch,args.max_epoch+1):
            # train
            # self.train_one_epoch(model=model,
            #                      optimizer=optimizer,
            #                      dataloader=train_loader,
            #                      device=device,
            #                      epoch=epoch
            #                      )
            # scheduler.step()
            # save model and validate
            args.val_freq = 1
            if epoch % args.val_freq == 0:
                # save_checkpoint(args, epoch, model, optimizer, scheduler)
                print("Validation begin.......")
                # self.evaluate_one_epoch(
                #     model=model,
                #     dataloader=val_loader,
                #     device=device, 
                #     epoch=epoch, 
                #     args=args)
                self.infer(model=model,
                           dataloader=val_loader,
                           device=device,
                           epoch=epoch, 
                           args=args)
                return
          
                

if __name__ == '__main__':
    opt = parse_option()
    torch.backends.cudnn.enabled = True
    # 启用cudnn的自动优化模式
    torch.backends.cudnn.benchmark = True
    # 设置cudnn的确定性模式，pytorch确保每次运行结果的输出都保持一致
    torch.backends.cudnn.deterministic = True

    trainer = Trainer()
    trainer.main(opt)