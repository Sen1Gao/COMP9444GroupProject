"""
Group: Team1024
File name: training_frameworkn.py
Author: Sen Gao
"""

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import utilities

class CustomDataSet(Dataset):
    def __init__(self, root_dir, folder_name, transforms):
        self.data_path=Path(root_dir)/Path(folder_name)
        self.color_image_data_path=self.data_path/Path("color_image")
        self.index_label_data_path=self.data_path/Path("index_label")
        self.color_images_path=[path for path in self.color_image_data_path.iterdir() if path.is_file() and path.suffix==".png"]
        self.index_labels_path=[]
        for path in self.color_images_path:
            self.index_labels_path.append(self.index_label_data_path/Path(path.name))
        self.transforms=transforms
        self.num_image=len(self.color_images_path)
        
        path_pairs=[[color,index] for color,index in zip(self.color_images_path,self.index_labels_path)]
        with ThreadPoolExecutor() as executor:
            result=list(executor.map(self.__load_color_and_index__,path_pairs))
        self.color_index_pairs=result

    def __load_color_and_index__(self,path_pair):
        color_image_path=path_pair[0]
        index_label_path=path_pair[1]
        color_image=cv2.imread(color_image_path.as_posix(),cv2.IMREAD_UNCHANGED)
        index_label=cv2.imread(index_label_path.as_posix(),cv2.IMREAD_UNCHANGED)
        return[color_image,index_label]

    def __len__(self):
        return self.num_image

    def __getitem__(self, index):
        color_image,index_label=self.color_index_pairs[index]

        return None,None