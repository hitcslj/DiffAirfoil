import matplotlib.pyplot as plt
from scipy.interpolate import splev,splprep
from scipy import optimize
import numpy as np
import torch
import os
from scipy.spatial.distance import pdist, squareform


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
          mean = torch.tensor([0.50194553,0.01158151]).to(data.device)
          std = torch.tensor([0.35423523,0.03827245]).to(data.device)
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