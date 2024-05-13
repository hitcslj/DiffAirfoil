import matplotlib.pyplot as plt
from scipy.interpolate import splev,splprep
from scipy import optimize
import numpy as np
import torch
import os
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

def norm(data,dtype='ndarray'):
    if dtype=='ndarray':
      mean = np.array([0.50194553,0.01158151])
      std = np.array([0.35423523,0.03827245])
      return (data-mean)/std * 2 -1 
    elif dtype=='tensor':
      mean = torch.tensor([0.50194553,0.01158151]).to(data.device)
      std = torch.tensor([0.35423523,0.03827245]).to(data.device)
      return (data-mean)/std * 2 -1
      
def de_norm(data,dtype='ndarray',dim=2):
    if dim==2:
      if dtype == 'ndarray':
          mean = np.array([0.50194553,0.01158151])
          std = np.array([0.35423523,0.03827245])
          return (data+1)/2 *std + mean
      elif dtype == 'tensor':
          mean = torch.tensor([0.50194553,0.01158151],dtype=torch.float32).to(data.device)
          std = torch.tensor([0.35423523,0.03827245],dtype=torch.float32).to(data.device)
          return (data+1)/2 *std + mean
    elif dim==1: # 给y de_norm
      if dtype == 'ndarray':
          mean = np.array([0.01158151])
          std = np.array([0.03827245])
          return (data+1)/2 *std + mean
      elif dtype == 'tensor':
          mean = torch.tensor([0.01158151]).to(data.device)
          std = torch.tensor([0.03827245]).to(data.device)
          return (data+1)/2 *std + mean
        

def vis_airfoil(data,idx,dir_name='output_airfoil'):
    os.makedirs(dir_name,exist_ok=True)
    ## 将data可视化，画出散点图
    plt.scatter(data[:,0],data[:,1])
     
    file_path = f'{dir_name}/{idx}.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()


def vis_airfoil2(source,target,idx,dir_name='output_airfoil',sample_type='ddpm'):
    os.makedirs(dir_name,exist_ok=True)
 
    ## 将source和target放到一张图
    plt.scatter(source[:, 0], source[:, 1], c='red', label='source')  # plot source points in red
    plt.scatter(target[:, 0], target[:, 1], c='blue', label='target')  # plot target points in blue
    plt.legend()  # show legend

    file_path = f'{dir_name}/{sample_type}_{idx}.png'
    plt.savefig(file_path, dpi=100, bbox_inches='tight', pad_inches=0.0)
    # Clear the plot cache
    plt.clf()

def vis_airfoils(airfoil_x, airfoil_y, epoch, dir_name='output_airfoil', title="Airfoil Plot"):
    os.makedirs(dir_name,exist_ok=True)
    idx = 0
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    for row in ax:
        for col in row:
            if idx >= len(airfoil_y):
                col.axis('off')
            else:
                y_plot = airfoil_y[idx].numpy() if isinstance(airfoil_y[idx], torch.Tensor) else airfoil_y[idx]
                col.scatter(airfoil_x, y_plot, s=0.6, c='black')
                col.axis('off')
                col.axis('equal')
                idx += 1
    file_path = f'{dir_name}/{epoch}.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def visualize_latent_space(vae, data_loader, device, epoch, dir_name='output_airfoil'):
    os.makedirs(dir_name,exist_ok=True)
    # Encode all data to get the latent vectors
    mu_list = []
    for data in data_loader:
        data = data['gt'][:,:,1] # [b,257]
        data = data.to(device)
        with torch.no_grad():
            mu, _ = vae.encode(data)
            mu_list.append(mu)
    
    # Concatenate all mu vectors from batches and remove any extra dimensions
    mu_tensor = torch.cat(mu_list, dim=0)
    mu_tensor = mu_tensor.view(mu_tensor.size(0), -1)  # Ensure it's a 2D array
    # Apply t-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, random_state=42)
    mu_tsne = tsne.fit_transform(mu_tensor.cpu().numpy())  # Ensure tensor is on CPU

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], alpha=0.5)
    plt.title('Latent Space Visualization using t-SNE')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    file_path = f'{dir_name}/{epoch}.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.clf()


def vis_traj(model, vae, val_dataset, log_dir, epoch, device="cuda:0"):
    n_samples = 1
    real_airfoils = torch.stack([val_dataset.__getitem__(i)['gt'][:,1] for i in range(n_samples)])
    y1 = torch.stack([val_dataset.__getitem__(i)['params'] for i in range(n_samples)]).to(device) # n, 1, 11
    y2 = torch.stack([val_dataset.__getitem__(i)['keypoint'][:,1] for i in range(n_samples)]).to(device)
    sample, all_samples = model.sampling(n_samples, y1, y2, device=device)
    all_samples = all_samples.squeeze()
    print("sample shape: ", sample.shape)

    gen_airfoils = vae.decode(all_samples).detach().cpu().numpy()
    gt_x = val_dataset.__getitem__(0)['gt'][:,0:1].cpu().numpy()
    print("generated_airfoils shape: ", gen_airfoils.shape)

    vis_airfoils(gt_x, np.expand_dims(real_airfoils.cpu().numpy(), -1),f"{epoch}_Real",dir_name=log_dir,title="Real Airfoils")
    vis_airfoils(gt_x, np.expand_dims(gen_airfoils, -1),f"{epoch}_Reconstructed",dir_name=log_dir,title="Traj of Airfoils")  

def eval(model, vae, val_dataset, log_dir, epoch=0, n_samples=16, device="cuda:0"):
    real_airfoils = torch.stack([val_dataset.__getitem__(i)['gt'][:,1] for i in range(n_samples)])
    y1 = torch.stack([val_dataset.__getitem__(i)['params'] for i in range(n_samples)]).to(device) # n, 1, 11
    y2 = torch.stack([val_dataset.__getitem__(i)['keypoint'][:,1] for i in range(n_samples)]).to(device)
    sample, all_samples = model.sampling(n_samples, y1, y2, device=device)
    sample = sample.squeeze()
    print("sample shape: ", sample.shape)
    
    gen_airfoils = vae.decode(sample).detach().cpu().numpy()
    gt_x = val_dataset.__getitem__(0)['gt'][:,0:1].cpu().numpy()
    print("generated_airfoils shape: ", gen_airfoils.shape)

    vis_airfoils(gt_x, np.expand_dims(real_airfoils.cpu().numpy(), -1),f"{epoch}_Real",dir_name=log_dir,title="Real Airfoils")
    vis_airfoils(gt_x, np.expand_dims(gen_airfoils, -1),f"{epoch}_Reconstructed",dir_name=log_dir,title="Reconstructed Airfoils")

def calculate_smoothness(airfoil):
    smoothness = 0.0
    num_points = airfoil.shape[0]

    for i in range(num_points):
        p_idx = (i - 1) % num_points
        q_idx = (i + 1) % num_points

        p = airfoil[p_idx]
        q = airfoil[q_idx]

        if p[0] == q[0]:  # 处理垂直于x轴的线段
            distance = abs(airfoil[i, 0] - p[0])
        else:
            m = (q[1] - p[1]) / (q[0] - p[0])
            b = p[1] - m * p[0]

            distance = abs(m * airfoil[i, 0] - airfoil[i, 1] + b) / np.sqrt(m**2 + 1)

        smoothness += distance

    return smoothness

def cal_diversity_score(data, subset_size=10, sample_times=1000):
    # Average log determinant
    N = data.shape[0]
    data = data.reshape(N, -1) 
    mean_logdet = 0
    for i in range(sample_times):
        ind = np.random.choice(N, size=subset_size, replace=False)
        subset = data[ind]
        D = squareform(pdist(subset, 'euclidean'))
        S = np.exp(-0.5*np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        mean_logdet += logdet
    return mean_logdet/sample_times


class Fit_airfoil():
    '''
    Fit airfoil by 3 order Bspline and extract Parsec features.
    airfoil (npoints,2)
    '''
    def __init__(self,airfoil,iLE=128):
        self.iLE = iLE
        self.tck, self.u  = splprep(airfoil.T,s=0)

        # parsec features
        # import pdb; pdb.set_trace()
        rle = self.get_rle()
        xup, yup, yxxup = self.get_up()
        xlo, ylo, yxxlo = self.get_lo()
        yteup = airfoil[0,1]
        ytelo = airfoil[-1,1]
        alphate, betate = self.get_te_angle()

        self.parsec_features = np.array([rle,xup,yup,yxxup,xlo,ylo,yxxlo,
                                         yteup,ytelo,alphate,betate]) 
        
        # 超临界翼型的特征
        xaft, yaft, yxxaft = self.get_aftload()
        # print(xaft, yaft, yxxaft)

    def get_rle(self):
        uLE = self.u[self.iLE]
        xu,yu = splev(uLE, self.tck,der=1) # dx/du
        xuu,yuu = splev(uLE, self.tck,der=2) # ddx/du^2
        K = abs(xu*yuu-xuu*yu)/(xu**2+yu**2)**1.5 # curvature
        return 1/K
    
    def get_up(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0,self.u[self.iLE]),method='bounded')
        uup = res.x
        xup ,yup = splev(uup, self.tck)

        xu,yu = splev(uup, self.tck, der=1) # dx/du
        xuu,yuu = splev(uup, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xup, yup, yxx

    def get_lo(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(self.u[self.iLE],1),method='bounded')
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1) # dx/du
        xuu,yuu = splev(ulo, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx

    def get_te_angle(self):
        xu,yu = splev(0, self.tck, der=1)
        yx = yu/xu
        alphate = np.arctan(yx)

        xu,yu = splev(1, self.tck, der=1)
        yx = yu/xu
        betate = np.arctan(yx)

        return alphate, betate
    
    # 后加载位置
    def get_aftload(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0.75,1),method='bounded')
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1) # dx/du
        xuu,yuu = splev(ulo, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx