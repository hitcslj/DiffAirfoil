import numpy as np
from scipy.interpolate import splev,splprep
from scipy import optimize
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import torch

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

def read_file(file_path):
    data = []
    with open(file_path) as file:
        for line in file:
            values = line.strip().split()
            data.append([float(values[0]), float(values[1])])
    return np.array(data)


def write_parsec_params(file_path, parsec_features):
    parsec_params_path = f'data/airfoil/parsec_params_direct_{dataset_name}.txt'
    with open(parsec_params_path, 'a') as f:
        f.write(file_path)
        f.write(',')
        f.write(','.join(map(str, parsec_features)))
        f.write('\n')


def process_file(file_path):
    data = read_file(file_path)
    f = Fit_airfoil(data)
    write_parsec_params(file_path, f.parsec_features)


if __name__ == '__main__':
    dataset_name = 'supercritical_airfoil'
    root_path = f'data/airfoil/{dataset_name}'
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    # 并行处理所有文件
    with Pool(processes=8) as pool:
        pool.map(process_file, file_paths)