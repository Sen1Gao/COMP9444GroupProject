"""
Group: Team1024
File name: index_label_generating.py
Author: Sen Gao
"""

import sys
from pathlib import Path

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from utilities import index_lookup,get_sub_dir_list,get_sub_file_list

def process_row(row:np.ndarray)->list:
    index_list=[]
    for element in row:
        bgr=element.tolist()
        rgb=(bgr[2],bgr[1],bgr[0])
        index=index_lookup(rgb)
        index_list.append(index)
    return index_list

def convert_color2index(file_path:Path)->np.ndarray|None:
    color=cv2.imread(file_path.as_posix(),cv2.IMREAD_COLOR)
    if color is None:
        return None

    rows=[color[i,:] for i in range(color.shape[0])]
    with ThreadPoolExecutor() as executor:
        rst=list(executor.map(process_row,rows))
    index=np.array(rst,dtype=np.uint8)
    return index

def process_color_label(file_path:Path,saving_path:Path)->bool:
    return cv2.imwrite(saving_path.as_posix(),convert_color2index(file_path))
    
if __name__ == "__main__":
    # Set directory of RUGD_annotations here
    annotations_path=Path("C:/Users/SenGao/Downloads/RUGD_annotations")
    if annotations_path.exists() and annotations_path.is_dir():
        annotation_index_name=annotations_path.name+"_index"
        annotation_index_path=annotations_path.parent/annotation_index_name
        if annotation_index_path.exists() and annotation_index_path.is_dir():
            print(f"'{annotation_index_path}' exists! Delete it before generating index labels.")
            sys.exit(0)
        annotation_index_path.mkdir()

        sub_dir_list=get_sub_dir_list(annotations_path)
        for p in sub_dir_list:
            new_p=Path(p.parent.parent/annotation_index_name/p.name)
            new_p.mkdir()
            sub_file_list=get_sub_file_list(p,".png")
            for f in tqdm(sub_file_list,desc=f"folder '{p.name}' is being processed.",total=len(sub_file_list)):
                process_color_label(f,new_p/f.name)
        print("Finish")
    else:
        print(f"{annotations_path} does not exist or is not a directory!")