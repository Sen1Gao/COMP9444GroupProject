"""
Group: Team1024
File name: index_label_generator.py
Author: Sen Gao
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import utilities

def process_row(img_row:np.ndarray)->list:
    index_row=[]
    for value in img_row:
        color_value_bgr=value.tolist()
        color_value_rgb=tuple([color_value_bgr[2],color_value_bgr[1],color_value_bgr[0]])
        index=utilities.index_lookup(color_value_rgb)
        index_row.append(index)

    return index_row

def convert_image(color_label_path:Path,index_label_path:Path)->bool:
    color_label=cv2.imread(color_label_path.as_posix(),cv2.IMREAD_UNCHANGED)
    if color_label is None:
        #print(f"Cannot load color label which is located in '{color_label_path.as_posix()}'!")
        return False
    
    h=color_label.shape[0]
    rows=[]
    for row in range(h):
        rows.append(color_label[row,:])
    
    with ThreadPoolExecutor() as executor:
        result=list(executor.map(process_row,rows))
    index_label=np.array(result,dtype=np.uint8)

    #Save converted index label
    if not cv2.imwrite(index_label_path.as_posix(),index_label):
        #print(f"Cannot save index label which should be located in '{index_label_path.as_posix()}'!")
        return False
    
    return True

def main(args=None):
    print("====================")
    color_label_folder_path=Path(args.input_folder_path.as_posix())
    print(f"Color label folder path: {color_label_folder_path.as_posix()}")

    try:
        print("Create a new folder to save generated index labels.")
        index_label_folder_path=color_label_folder_path.parent/(color_label_folder_path.name+"_index")
        index_label_folder_path.mkdir(exist_ok=False)

        sub_color_folder_paths=[path for path in color_label_folder_path.iterdir() if path.is_dir()]
        color_label_file_paths=[]
        index_label_file_paths=[]
        for sub_color_folder_path in sub_color_folder_paths:
            sub_index_folder_path=index_label_folder_path/sub_color_folder_path.name
            sub_index_folder_path.mkdir(exist_ok=False)
            sub_color_folder_file_paths=[file for file in sub_color_folder_path.iterdir() if file.is_file() and file.suffix==".png"]
            color_label_file_paths.extend(sub_color_folder_file_paths)
            for sub_color_folder_file_path in sub_color_folder_file_paths:
                index_label_file_paths.append(sub_index_folder_path/sub_color_folder_file_path.name)
        print(f"Folder '{index_label_folder_path.as_posix()}' and its sub folder have been created successfully.")

        print("Start generating index labels.")
        for color_label_file_path,index_label_file_path in tqdm(zip(color_label_file_paths,index_label_file_paths),total=len(color_label_file_paths)):
            convert_image(color_label_file_path,index_label_file_path)
        print("Finish.")
        print("====================")

    except Exception as e:
        print(e)
        sys.exit(0)

    
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_folder_path",
                        type=Path,
                        required=True,
                        help="The path of folder of annotation. For example: --input_folder_path your/RUGD_annotations/folder/path")
    args=parser.parse_args()
    
    if not args.input_folder_path.exists() or not args.input_folder_path.is_dir():
        print("The folder path inputted does not exist or it is not a folder path!")
        sys.exit(0)

    main(args)
