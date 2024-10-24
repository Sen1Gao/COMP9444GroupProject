"""
Group: Team1024
File name: dataset_loading.py
Author: Sen Gao
"""

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir:str, folder_name:str, color_image_transform:transforms, index_label_transform:transforms, target_size=None|tuple):
        self.data_path=Path(root_dir)/Path(folder_name)
        self.color_image_data_path=self.data_path/Path("color_image")
        self.index_label_data_path=self.data_path/Path("index_label")
        self.color_images_path=[path for path in self.color_image_data_path.iterdir() if path.is_file() and path.suffix==".png"]
        self.index_labels_path=[]
        for path in self.color_images_path:
            self.index_labels_path.append(self.index_label_data_path/Path(path.name))
        self.color_image_transform=color_image_transform
        self.index_label_transform=index_label_transform
        self.target_size=target_size
        self.num_image=len(self.color_images_path)
        
        path_pairs=[[color,index] for color,index in zip(self.color_images_path,self.index_labels_path)]
        with ThreadPoolExecutor() as executor:
            result=list(tqdm(executor.map(self.__load_color_and_index__,path_pairs),desc="Load images and labels",total=self.__len__()))
        self.color_index_pairs=result
        print("the split dataset has been loaded.")

    def __load_color_and_index__(self,path_pair)->list:
        color_image_path=path_pair[0]
        index_label_path=path_pair[1]
        color_image=cv2.imread(color_image_path.as_posix(),cv2.IMREAD_UNCHANGED)
        index_label=cv2.imread(index_label_path.as_posix(),cv2.IMREAD_UNCHANGED)
        if self.target_size is not None:
            color_image=cv2.resize(color_image,self.target_size,interpolation=cv2.INTER_LINEAR)
            index_label=cv2.resize(index_label,self.target_size,interpolation=cv2.INTER_NEAREST)
        index_label=index_label.astype(np.int8)
        return[color_image,index_label]

    def __len__(self)->int:
        return self.num_image

    def __getitem__(self, index):
        color_image,index_label=self.color_index_pairs[index]
        color_image=self.color_image_transform(color_image)
        index_label=self.index_label_transform(index_label)
        index_label=torch.from_numpy(index_label).long() 
        return color_image,index_label

