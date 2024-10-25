"""
Group: Team1024
File name: training_frameworkn.py
Author: Sen Gao
"""

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import utilities
from dataset_loading import CustomDataset
from Networks import U_Net

class Trainer():
    def __init__(self,classifier,class_num,is_printing_model=False):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Current used device is {self.device}")
        self.class_num=class_num
        self.model=classifier(class_num)
        if is_printing_model is True:
            print("Model Information:")
            print(self.model)
        self.device_model=self.model.to(self.device)
        self.train_data=None
        self.val_data=None
        self.test_data=None
        self.learning_rate=None
        self.batch_size=None
        self.epochs=None
        self.ignore_index=None
        
    def __update_iou__(self,preds, labels, total_intersection,total_union):
        for pred, label in zip(preds, labels):
            for index in range(self.class_num):
                pred_inds = (pred == index)
                label_inds = (label == index)
                intersection = (pred_inds & label_inds).sum().item()
                union = (pred_inds | label_inds).sum().item()
                total_intersection[index] += intersection
                total_union[index] += union
        return total_intersection,total_union
        
    def load_image_data(self,root_dir,color_transform,index_transform,target_size=None):
        self.train_data = CustomDataset(root_dir, 'train', color_transform,index_transform, target_size)
        self.val_data = CustomDataset(root_dir, 'val',color_transform, index_transform,target_size)
        self.test_data = CustomDataset(root_dir, 'test',color_transform,index_transform, target_size)
    
    def set_training_parameters(self,learning_rate=0.01,batch_size=16,epochs=50,ignore_index=-100):
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.epochs=epochs
        self.ignore_index=ignore_index

    def start_training(self,model_save_path):
        y_training_loss = np.array([0]*self.epochs, dtype=float)
        y_val_loss = np.array([0]*self.epochs, dtype=float)
        y_miou = np.array([0]*self.epochs, dtype=float)
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        optimizer = optim.Adam(self.device_model.parameters(), lr=self.learning_rate)
        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True,drop_last=True)
        val_loader = DataLoader(self.val_data, self.batch_size, shuffle=True,drop_last=True)
        torch.cuda.empty_cache()
        for epoch in range(self.epochs):
            message = f"Current epoch:{epoch+1}/{self.epochs} Training"
            self.device_model.train()
            batch_training_loss_sum = 0.0
            for images, labels in tqdm(train_loader, desc=message, leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.device_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_training_loss_sum += loss.item()
            avg_training_loss = batch_training_loss_sum / len(train_loader)
            y_training_loss[epoch] = avg_training_loss

            message = f"Current epoch:{epoch+1}/{self.epochs} validating"
            self.device_model.eval()
            batch_val_loss_sum = 0.0
            mIoU=0.0
            total_intersection = np.array([0]*self.class_num,dtype=float)
            total_union = np.array([0]*self.class_num,dtype=float)
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=message, leave=False):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.device_model(images)
                    loss = criterion(outputs, labels)
                    batch_val_loss_sum += loss.item()
                    pred_labels = torch.argmax(outputs, dim=1)
                    pred_labels_np = pred_labels.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    total_intersection,total_union=self.__update_iou__(pred_labels_np,labels_np,total_intersection,total_union)
            iou=total_intersection[1:]/(total_union+1e-6)[1:]
            mIoU=iou.mean().item()
            avg_val_loss = batch_val_loss_sum/len(val_loader)
            y_val_loss[epoch] = avg_val_loss
            y_miou[epoch]=mIoU
            print(f"Epoch:{epoch+1}/{self.epochs} Training loss: {avg_training_loss:.4f} Validation loss: {avg_val_loss:.4f} mIoU of Validation: {mIoU:.4f}")
        self.save_current_model(model_save_path)
        return y_training_loss,y_val_loss,y_miou
    
    def draw_training_result(self,train_loss,val_loss,mIoU):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, color='blue', label='Traning loss')
        plt.plot(val_loss, color='orange', label='Validation loss')
        plt.title("Loss")
        plt.legend(loc='best')
        plt.subplot(1, 2, 2)
        plt.plot(mIoU, color='green', label='mIou')
        plt.legend(loc='best')
        plt.title("Mean Iou")
        plt.show()
        
    def start_testing(self):
        torch.cuda.empty_cache()
        test_loader = DataLoader(self.test_data, self.batch_size, shuffle=False,drop_last=True)
        self.device_model.eval()
        mIoU=0.0
        total_intersection = np.array([0]*self.class_num,dtype=float)
        total_union = np.array([0]*self.class_num,dtype=float)
        with torch.no_grad():
            message = "Test"
            for images, labels in tqdm(test_loader, desc=message, leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.device_model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                total_intersection,total_union=self.__update_iou__(preds_np,labels_np,total_intersection,total_union)
        iou=total_intersection[1:]/(total_union+1e-6)[1:]
        mIoU=iou.mean().item()
        print(f"mIoU of test set:{mIoU}")
    
    def save_current_model(self,model_save_path):
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime(f"%Y-%m-%d-%H-%M-%S")
        scripted_model = torch.jit.script(self.device_model)
        scripted_model.save(f"{model_save_path}/{timestamp_str}_model.pt")


color_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4013, 0.4035, 0.4035],std=[0.2779, 0.2754, 0.2736])
])
label_transform = transforms.Compose([
    
])


trainer=Trainer(U_Net.UNet,25)

data_set_path="C:/Users/SenGao/Downloads/RUGD_split_dataset"
trainer.load_image_data(data_set_path, color_transform,label_transform,(512,512))

trainer.set_training_parameters(0.01,5,40)

model_save_path=data_set_path
y1,y2,y3=trainer.start_training(model_save_path)

trainer.draw_training_result(y1, y2, y3)

trainer.start_testing()