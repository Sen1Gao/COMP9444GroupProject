"""
Group: Team1024
File name: standardization_para_calculation.py
Author: Sen Gao
"""

from pathlib import Path

from tqdm import tqdm
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader

class CustomDataset(Dataset):
    def __init__(self,data_dir) -> None:
        self.data_dir_path=Path(data_dir)
        self.image_paths=[]
        if self.data_dir_path.exists() and self.data_dir_path.is_dir():
            self.image_paths.extend([path for path in self.data_dir_path.iterdir() if path.is_file() and path.suffix==".png"])
        self.image_paths_length=len(self.image_paths)

    def __len__(self) -> int:
        return self.image_paths_length
    
    def __getitem__(self, index):
        image=cv2.imread(self.image_paths[index].as_posix(),cv2.IMREAD_UNCHANGED)
        image=transforms.ToTensor()(image)
        return image

def calculate_standardization_para(data_dir:str,batch_size=1,num_works=1):
    dataset=CustomDataset(data_dir=data_dir)
    data_loader=DataLoader(dataset=dataset,batch_size=batch_size,num_workers=num_works)

    sum_bgr=torch.zeros(3,dtype=torch.float64)
    num_pixel=0
    for images in tqdm(data_loader,desc="Calculate mean"):
        sum_bgr+=torch.sum(images,[0,2,3]).double()
        num_pixel+=images.size(0)*images.size(2)*images.size(3)
    mean=(sum_bgr/num_pixel).float()
    print(f"mean: {mean}")

    sum_squared_bgr=torch.zeros(3,dtype=torch.float64)
    for images in tqdm(data_loader,desc="Calculate std"):
        resharped_mean=mean[None, :, None, None]
        difference_value=images-resharped_mean
        squared_difference_value=difference_value**2
        sum_squared_bgr+=squared_difference_value.sum([0,2,3]).double()
    std=torch.sqrt(sum_squared_bgr/num_pixel).float()
    print(f"std: {std}")


if __name__ == "__main__":
    calculate_standardization_para(data_dir="/RUGD_split_dataset/train/color_image",
                                   batch_size=10,
                                   num_works=6)
