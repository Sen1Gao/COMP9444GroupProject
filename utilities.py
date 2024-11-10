from pathlib import Path

import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

# RGB format
index_color_mapping={0:(0,0,0),         # void
                     1:(108,64,20),     # dirt
                     2:(255,229,204),   # sand
                     3:(0,102,0),       # grass
                     4:(0,255,0),       # tree
                     5:(0,153,153),     # pole
                     6:(0,128,255),     # water
                     7:(0,0,255),       # sky
                     8:(255,255,0),     # vehicle
                     9:(255,0,127),     # container/generic-object
                     10:(64,64,64),     # asphalt
                     11:(255,128,0),    # gravel
                     12:(255,0,0),      # building
                     13:(153,76,0),     # mulch
                     14:(102,102,0),    # rock-bed
                     15:(102,0,0),      # log
                     16:(0,255,128),    # bicycle
                     17:(204,153,255),  # person
                     18:(102,0,204),    # fence
                     19:(255,153,204),  # bush
                     20:(0,102,102),    # sign
                     21:(153,204,255),  # rock
                     22:(102,255,255),  # bridge
                     23:(101,101,11),   # concrete
                     24:(114,85,47)}    # picnic-table

# RGB format
color_index_mapping={(0,0,0):0,         # void
                     (108,64,20):1,     # dirt
                     (255,229,204):2,   # sand
                     (0,102,0):3,       # grass
                     (0,255,0):4,       # tree
                     (0,153,153):5,     # pole
                     (0,128,255):6,     # water
                     (0,0,255):7,       # sky
                     (255,255,0):8,     # vehicle
                     (255,0,127):9,     # container/generic-object
                     (64,64,64):10,     # asphalt
                     (255,128,0):11,    # gravel
                     (255,0,0):12,      # building
                     (153,76,0):13,     # mulch
                     (102,102,0):14,    # rock-bed
                     (102,0,0):15,      # log
                     (0,255,128):16,    # bicycle
                     (204,153,255):17,  # person
                     (102,0,204):18,    # fence
                     (255,153,204):19,  # bush
                     (0,102,102):20,    # sign
                     (153,204,255):21,  # rock
                     (102,255,255):22,  # bridge
                     (101,101,11):23,   # concrete
                     (114,85,47):24}    # picnic-table     

def index_lookup(color:tuple)->int:
    """
    Get index of color from color_index_mapping where the format of color is RGB.\n
    Therefore, you must convert color format to RGB before you pass variable 'color'\n
    The variable 'color' is a tuple.
    """
    return color_index_mapping[color]

def to_color_label(index_label:np.ndarray) -> np.ndarray:
    """
    Convert index label to color label for showing the result of prediction.
    """
    h,w=index_label.shape
    color_label=np.zeros((h,w,3),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            color=index_color_mapping[index_label[i][j]]
            r,g,b=color
            color_label[i][j]=np.array([b,g,r])
    return color_label

def get_dirs_list(dir_path:Path)->list:
    return [p for p in dir_path.iterdir() if p.is_dir()]

def get_files_list(dir_path:Path,suffix:str)->list:
    return [f for f in dir_path.iterdir() if f.is_file() and f.suffix==suffix]

def process_row(row:np.ndarray)->list:
    index_list=[]
    for element in row:
        bgr=element.tolist()
        rgb=(bgr[2],bgr[1],bgr[0])
        index_list.append(index_lookup(rgb))
    return index_list

def convert_color2index(file_path:str)->np.ndarray:
    color=cv2.imread(file_path,cv2.IMREAD_COLOR)
    rows=[color[i,:] for i in range(color.shape[0])]
    with ThreadPoolExecutor() as executor:
        rst=list(executor.map(process_row,rows))
    return np.array(rst,dtype=np.uint8)