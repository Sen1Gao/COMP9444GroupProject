"""
Group: Team1024
File name: dataset_spliting.py
Author: Sen Gao
"""

from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_sub_dir_list(parent_path:Path)->list|None:
    if parent_path.exists() and parent_path.is_dir():
        return [p for p in parent_path.iterdir() if p.is_dir()]
    return None

def get_sub_file_list(parent_path:Path,suffix:str)->list|None:
    if parent_path.exists() and parent_path.is_dir():
        return [f for f in parent_path.iterdir() if f.is_file() and f.suffix==suffix]
    return None

def load_label(label_path:Path):
    return cv2.imread(label_path.as_posix(),cv2.IMREAD_UNCHANGED)

def compute_pixel_distribution(labels, num_classes):
    N = labels.shape[0]
    pixel_distribution = np.zeros((N, num_classes), dtype=np.float32)

    for i in tqdm(range(N),desc=f"Calculate distribution",total=N):
        unique, counts = np.unique(labels[i], return_counts=True)
        total_pixels = labels[i].size
        for u, count in zip(unique, counts):
            pixel_distribution[i, u] = count / total_pixels

    return pixel_distribution

def stratified_split(labels, num_classes, test_size=0.2):
    pixel_distribution = compute_pixel_distribution(labels, num_classes)

    overall_distribution = np.mean(pixel_distribution, axis=0)

    train_indices, test_indices = [], []

    for i, dist in enumerate(pixel_distribution):
        if np.random.rand() > test_size:
            train_indices.append(i)
        else:
            test_indices.append(i)

    train_pixel_distribution = np.mean(pixel_distribution[train_indices], axis=0)
    test_pixel_distribution = np.mean(pixel_distribution[test_indices], axis=0)
    
    x=np.arange(0,num_classes,1)
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1,3,1)
    plt.plot(x, train_pixel_distribution, color='blue', label='train')
    plt.legend(loc='best')
    plt.subplot(1,3,2)
    plt.plot(x, test_pixel_distribution, color='blue', label='test')
    plt.legend(loc='best')
    plt.subplot(1,3,3)
    plt.plot(x, overall_distribution, color='blue', label='overall')
    plt.legend(loc='best')
    plt.show()

    return (train_indices,test_indices)

if __name__ == "__main__":
    # Set directory of RUGD_frames-with-annotations here
    frames_path=Path("C:/Users/SenGao/Downloads/RUGD_frames-with-annotations")
    # Set directory of RUGD_annotations_index here
    annotations_index_path=Path("C:/Users/SenGao/Downloads/RUGD_annotations_index")
    # Set test data ratio here
    test_size=0.2
    # Set class number here
    num_class=25
    # Set directory of spliting result here
    saving_path=Path("C:/Users/SenGao/Downloads")

    if frames_path.exists() and frames_path.is_dir() and annotations_index_path.exists() and annotations_index_path.is_dir():
        y=[]
        X_path_list=[]
        y_path_list=[]
        frame_sub_dir_list=get_sub_dir_list(frames_path)
        for p in frame_sub_dir_list:
            frame_sub_file_list=get_sub_file_list(p,".png")
            for f in tqdm(frame_sub_file_list,desc=f"Load images and labels from '{p.name}'.",total=len(frame_sub_file_list)):
                frame_path=f
                label_path=annotations_index_path/frame_path.parent.name/frame_path.name
                X_path_list.append(frame_path)
                y_path_list.append(label_path)
                label=load_label(label_path)
                y.append(label)
        
        y=np.array(y)
        (train_indices,test_indices) = stratified_split(y, num_class, test_size)

        train_data_path_str_list=[]
        for index in train_indices:
            index_frame_path=Path("")/X_path_list[index].parent.parent.name/X_path_list[index].parent.name/X_path_list[index].name
            index_label_path=Path("")/y_path_list[index].parent.parent.name/y_path_list[index].parent.name/y_path_list[index].name
            train_data_path_str_list.append([index_frame_path.as_posix(),index_label_path.as_posix()])
        np.savetxt((saving_path/"train_data_path.csv").as_posix(),np.array(train_data_path_str_list),delimiter=',',fmt='%s')
        test_data_path_str_list=[]
        for index in test_indices:
            index_frame_path=Path("")/X_path_list[index].parent.parent.name/X_path_list[index].parent.name/X_path_list[index].name
            index_label_path=Path("")/y_path_list[index].parent.parent.name/y_path_list[index].parent.name/y_path_list[index].name
            test_data_path_str_list.append([index_frame_path.as_posix(),index_label_path.as_posix()])
        np.savetxt((saving_path/"test_data_path.csv").as_posix(),np.array(test_data_path_str_list),delimiter=',',fmt='%s')

        print("Finish")
    else:
        print(f"{frames_path} or {annotations_index_path} does not exist or is not a directory!")