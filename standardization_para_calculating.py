"""
Group: Team1024
File name: standardization_para_calculation.py
Author: Sen Gao
"""

from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

class CustomDataset(Dataset):
    def __init__(self,train_data_root_dir,train_data_csv_file_name) -> None:
        train_data_csv_file_path=Path(train_data_root_dir)/train_data_csv_file_name
        train_data_file_dir_list= np.genfromtxt(train_data_csv_file_path,dtype=None,encoding='utf-8',delimiter=',')
        self.frame_path_list=[]
        for s in train_data_file_dir_list[:,:1].tolist():
            self.frame_path_list.append(Path(train_data_root_dir)/s[0])

    def __len__(self) -> int:
        return len(self.frame_path_list)
    
    def __getitem__(self, index):
        image=cv2.imread(self.frame_path_list[index].as_posix(),cv2.IMREAD_UNCHANGED)
        image=transforms.ToTensor()(image)
        return image
    


if __name__ == "__main__":
    train_data_root_dir="C:/Users/SenGao/Downloads"
    train_data_csv_file_name="train_data_path.csv"
    dataset=CustomDataset(train_data_root_dir,train_data_csv_file_name)
    data_loader=DataLoader(dataset=dataset,batch_size=32,num_workers=6)

    sum_bgr=torch.zeros(3,dtype=torch.float64)
    num_pixel=0
    for images in tqdm(data_loader,desc="Calculate mean"):
        sum_bgr+=torch.sum(images,[0,2,3])
        num_pixel+=images.size(0)*images.size(2)*images.size(3)
    mean=(sum_bgr/num_pixel)
    print(f"mean: {mean}")

    sum_squared_bgr=torch.zeros(3,dtype=torch.float64)
    for images in tqdm(data_loader,desc="Calculate std"):
        resharped_mean=mean[None, :, None, None]
        difference_value=images-resharped_mean
        squared_difference_value=difference_value**2
        sum_squared_bgr+=squared_difference_value.sum([0,2,3])
    std=torch.sqrt(sum_squared_bgr/num_pixel)
    print(f"std: {std}")
