import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import utilities

def load_image_and_label(image_path,label_path,target_size):
    image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, target_size,interpolation=cv2.INTER_LINEAR)
    label=cv2.imread(label_path,cv2.IMREAD_UNCHANGED)
    label = cv2.resize(label, target_size,interpolation=cv2.INTER_NEAREST)
    return image,label

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_folder_path="C:/Users/SenGao/Downloads/RUGD_split_dataset"
model_name="2024-10-25-14-35-59_model.pt"
loaded_model = torch.jit.load(f"{model_folder_path}/{model_name}")
loaded_model.to(device)
loaded_model.eval()

image_path="C:/Users/SenGao/Downloads/RUGD_split_dataset/test/color_image/trail-7_00666.png"
label_path=image_path.replace("color_image", "index_label")
target_size=(512,512)
class_num=25

with torch.no_grad():
    image,label=load_image_and_label(image_path,label_path,target_size)
    image_tensor=transforms.ToTensor()(image).unsqueeze(0)
    output = loaded_model(image_tensor.to(device))
    prob = torch.softmax(output, dim=1)
    pred = torch.argmax(prob, dim=1)
    pred_np = pred.cpu().numpy().astype(np.uint8)
    label_color=utilities.to_color_label(label)
    pred_color=utilities.to_color_label(pred_np[0])
    
    cv2.imshow("image",image)
    cv2.waitKey(0)
    cv2.imshow("label_color",label_color)
    cv2.waitKey(0)
    cv2.imshow("pred_color",pred_color)
    cv2.waitKey(0)

