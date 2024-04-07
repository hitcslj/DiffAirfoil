import os
import torch
from torch.utils.data import Dataset
import numpy as np
from utils import norm

def get_data(txt_path):
    data = []
    with open(txt_path) as file:
        # 逐行读取文件内容
        for line in file:
            # 移除行尾的换行符，并将每行拆分为两个数值
            values = line.strip().split()
            # 将数值转换为浮点数，并添加到数据列表中
            data.append([float(values[0]), float(values[1])])
    data = torch.FloatTensor(data)
    return data



class AirFoilDatasetParsec(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 dataset_names = ['supercritical_airfoil'],
                 ):
        self.split = split
        txt_list = []
        for dataset_name in dataset_names:
            with open(f'data/airfoil/{dataset_name}_{split}.txt') as f:
                  txt_list += [os.path.join(f'data/airfoil/{dataset_name}',line.rstrip().strip('\n') + '.dat',) 
                              for line in f.readlines()]
        self.txt_list = txt_list
        params = {}
        for dataset_name in dataset_names:
          with open(f'data/airfoil/parsec_params_direct_{dataset_name}.txt') as f:
              for line in f.readlines():
                  name_params = line.rstrip().strip('\n').split(',')
                  # 取出路径的最后一个文件名作为key
                  name = name_params[0].split('/')[-1].split('.')[0]
                  params[name] = list(map(float,name_params[1:]))
        self.params = params
    
    def __getitem__(self, index):
        """Get current batch for input index"""
        txt_path = self.txt_list[index]
        key = txt_path.split('/')[-1].split('.')[0]
        params = self.params[key]
        data = get_data(txt_path)
        input = data[::10] # 25个点
        data = norm(data,dtype='tensor') # 归一化
        params = torch.FloatTensor(params)
        return {'keypoint':input,'gt':data,'params':params}
    
    def __len__(self):
        return len(self.txt_list)

class EditingDataset(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 datapath = 'data/airfoil/supercritical_airfoil',
                 ):
        self.split = split
        self.datapath = datapath
        
        with open('data/airfoil/%s.txt' % split) as f:
              txt_list = [os.path.join(datapath,line.rstrip().strip('\n') + '.dat',) 
                          for line in f.readlines()]
        self.txt_list = txt_list
        self.params = {}
        # params = []
        with open('data/airfoil/parsec_params_direct.txt') as f:
            for line in f.readlines():
                name_params = line.rstrip().strip('\n').split(',')
                # 取出路径的最后一个文件名作为key
                name = name_params[0].split('/')[-1].split('.')[0]
                self.params[name] = list(map(float,name_params[1:]))

    def __getitem__(self, index):
        """Get current batch for input index"""
        index2 = np.random.randint(len(self.txt_list))
        txt_path1 = self.txt_list[index]
        txt_path2 = self.txt_list[index2]
        key1 = txt_path1.split('/')[-1].split('.')[0]
        key2 = txt_path2.split('/')[-1].split('.')[0]
        params1 = torch.FloatTensor(self.params[key1])
        params2 = torch.FloatTensor(self.params[key2])
        source = get_data(txt_path1)
        target = get_data(txt_path2)
        source = norm(source)
        target = norm(target)
        source_keypoint = source[::10] # 26个点
        target_keypoint = target[::10] # 26个点
        return {'source_keypoint':source_keypoint,'source_point':source,'source_param':params1,
                'target_keypoint':target_keypoint,'target_point':target,'target_param':params2}
    
    def __len__(self):
        return len(self.txt_list)


class AirFoilMixParsec(Dataset):
    """Dataset for shape datasets(coco & 机翼)"""
    def __init__(self,split = 'train',
                 dataset_names = ['r05','r06', 'supercritical_airfoil'],
                 ):
        self.split = split
        txt_list = []
        for dataset_name in dataset_names:
            with open(f'data/airfoil/{dataset_name}_{split}.txt') as f:
                  txt_list += [os.path.join(f'data/airfoil/{dataset_name}',line.rstrip().strip('\n') + '.dat',) 
                              for line in f.readlines()]
        self.txt_list = txt_list
        params = {}
        for dataset_name in dataset_names:
          with open(f'data/airfoil/parsec_params_direct_{dataset_name}.txt') as f:
              for line in f.readlines():
                  name_params = line.rstrip().strip('\n').split(',')
                  # 取出路径的最后一个文件名作为key
                  name = name_params[0].split('/')[-1].split('.')[0]
                  params[name] = list(map(float,name_params[1:]))
        self.params = params
    
    def __getitem__(self, index):
        """Get current batch for input index"""
        txt_path = self.txt_list[index]
        key = txt_path.split('/')[-1].split('.')[0]
        params = self.params[key]
        data = get_data(txt_path)
        input = data[::10] # 25个点
        data = norm(data,dtype='tensor') # 归一化
        params = torch.FloatTensor(params)
        return {'keypoint':input,'gt':data,'params':params}
    
    def __len__(self):
        return len(self.txt_list)

if __name__ == '__main__':
    dataset = AirFoilMixParsec()
    print(dataset[0])