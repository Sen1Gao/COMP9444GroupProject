"""
Group: Team1024
File name: dataset_spliting.py
Author: Sen Gao
"""

from pathlib import Path
import shutil
import random

root_dir="C:/Users/SenGao/Downloads"
split_dataset_dir_name="RUGD_split_dataset"
traing_rate=0.8
val_rate=0.1
test_rate=0.1
random.seed(0)

image_dir_name="RUGD_frames-with-annotations"
label_dir_name="RUGD_annotations_index"
image_dir_path=Path(root_dir)/Path(image_dir_name)

training_files_path=[]
val_files_path=[]
test_files_path=[]

if image_dir_path.exists() and image_dir_path.is_dir() and traing_rate+val_rate+test_rate==1.0:
    sub_image_dirs_path=[dir for dir in image_dir_path.iterdir() if dir.is_dir()]
    if len(sub_image_dirs_path)>0:
        for dir_path in sub_image_dirs_path:
            image_files_path=[file for file in dir_path.iterdir() if file.is_file() and file.suffix==".png"]
            if len(image_files_path)>0:
                random_numbers=random.sample(range(len(image_files_path)),len(image_files_path))
                num_training_image=int(len(image_files_path)*traing_rate)
                num_val_image=int(len(image_files_path)*val_rate)
                num_test_image=int(len(image_files_path)-num_training_image-num_val_image)
                for i in range(num_training_image):
                    training_files_path.append(image_files_path[random_numbers[i]])
                for i in range(num_val_image):
                    val_files_path.append(image_files_path[random_numbers[i+num_training_image]])
                for i in range(num_test_image):
                    test_files_path.append(image_files_path[random_numbers[i+num_training_image+num_val_image]])
    
    if len(training_files_path)>0 and len(val_files_path)>0 and len(test_files_path)>0:
        split_dataset_dir_path=Path(root_dir)/Path(split_dataset_dir_name)
        if not split_dataset_dir_path.exists():
            split_dataset_dir_path.mkdir()
            for folder_name,files_path in zip(["train","val","test"],[training_files_path,val_files_path,test_files_path]):
                (split_dataset_dir_path/folder_name).mkdir()
                color_image_dir_path=split_dataset_dir_path/folder_name/Path("color_image")
                color_image_dir_path.mkdir()
                index_label_dir_path=split_dataset_dir_path/folder_name/Path("index_label")
                index_label_dir_path.mkdir()
                for file_path in files_path:
                    file_name=file_path.name
                    source=file_path.as_posix()
                    dest=(color_image_dir_path/Path(file_name)).as_posix()
                    shutil.copyfile(source,dest)

                    source=str.replace(source,image_dir_name,label_dir_name)
                    dest=(index_label_dir_path/Path(file_name)).as_posix()
                    shutil.copyfile(source,dest)
            print("Finish.")
        else:
            print(f"{split_dataset_dir_path.as_posix()} is already existed.")
            