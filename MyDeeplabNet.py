import torch
import torch.nn as nn
from torchvision import models
from timm.models.vision_transformer import VisionTransformer
import cv2
from torchvision import transforms
import numpy as np
import utilities




# class MyDeeplabV3Net(nn.Module):

#     def __init__(self,num_classes,atrous_rate):
#         super(MyDeeplabV3Net, self).__init__()
#         resnet=models.resnet50(pretrained=True)
#         self.backbone=nn.Sequential(
#             resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
#             resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
#         )

#         self.atrous_blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(2048, 256, kernel_size=3, padding=rate, dilation=rate, bias=False),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(inplace=True)
#             ) for rate in atrous_rate #(6, 12, 18)
#         ])


#         self.transformer = VisionTransformer(
#             img_size=16,
#             patch_size=4,
#             in_chans=2048,
#             embed_dim=256,
#             depth=6,
#             num_heads=8,
#             mlp_ratio=4,
#             num_classes=0 
#         )

#         self.output_conv = nn.Sequential(
#             nn.Conv2d(len(atrous_rate) * 256 + 256, 256, kernel_size=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True)
#         )

#         self.classifier = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, num_classes, kernel_size=1)
#         )

#     def forward(self, x):
#         x_out=self.backbone(x)
#         atrous_outs = [block(x_out) for block in self.atrous_blocks]
#         transformer_feat = self.transformer(x_out).view(x_out.shape[0],256,1,1)
#         transformer_feat = nn.functional.interpolate(transformer_feat, size=x_out.shape[2:], mode='bilinear', align_corners=True)
#         out = torch.cat(atrous_outs + [transformer_feat], dim=1)
#         out=self.output_conv(out)
#         out=self.classifier(out)
#         out = nn.functional.interpolate(out, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
#         return out
    
#model=MyDeeplabV3Net(25,(6, 12, 18))
# model = models.segmentation.deeplabv3_resnet50(pretrained=True)
# print(model)
# model.classifier[4] = nn.Conv2d(256, 25, kernel_size=(1, 1), stride=(1, 1))  # 替换为21类（以VOC为例）

# X=torch.randn(6,3,512,512)
# output=model(X)
# pass
#import NN
#from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
import BiSeNet

def load_image_and_label(image_path,label_path,target_size):
    image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size,interpolation=cv2.INTER_LINEAR)
    label=cv2.imread(label_path,cv2.IMREAD_UNCHANGED)
    label = cv2.resize(label, target_size,interpolation=cv2.INTER_NEAREST)
    return image,label

#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
print(device)
# model = models.segmentation.deeplabv3_resnet50(pretrained=True)
# model.classifier[4] = nn.Conv2d(256, 25, kernel_size=(1, 1), stride=(1, 1))
#model=NN.PSPNetWithDeepSupervision(25)
# model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")
# model.decode_head.classifier=nn.Conv2d(512, 25, kernel_size=(1, 1), stride=(1, 1))
# model.auxiliary_head.classifier=nn.Conv2d(256, 25, kernel_size=(1, 1), stride=(1, 1))
model=BiSeNet.BiSeNet(25)
print(model)
loaded_model = model#MyDeeplabV3Net(25,(6, 12, 18)) 
loaded_model.load_state_dict(torch.load("C:/Users/SenGao/Downloads/RUGD_ws/model/model_params_2_14.pth"))
loaded_model.to(device)
loaded_model.eval()

image_path="C:/Users/SenGao/Downloads/RUGD_ws/RUGD_frames-with-annotations/trail-3/trail-3_00596.png"
label_path=image_path.replace("RUGD_frames-with-annotations", "RUGD_annotations_index")
target_size=(512,512)
class_num=25

correct_pixel_sum=0
pixel_sum=0
total_intersection=np.zeros(class_num,dtype=np.int64)
total_union=np.zeros(class_num,dtype=np.int64)
with torch.no_grad():
    image,label=load_image_and_label(image_path,label_path,target_size)
    image_tensor=transforms.ToTensor()(image).unsqueeze(0)
    output= loaded_model(image_tensor.to(device))
    #prob = torch.softmax(output['out'], dim=1)
    prob = torch.softmax(output, dim=1)
    pred = torch.argmax(prob, dim=1)
    pred_np = pred.cpu().numpy().reshape(512,512).astype(np.uint8)
    correct_pixel_sum+=(pred_np==label).sum().item()
    pixel_sum+=label.size

pixel_accuracy=correct_pixel_sum/pixel_sum
print(pixel_accuracy)
cv2.imshow("image",image)
cv2.imshow("GT",utilities.to_color_label(label))
cv2.imshow("pred",utilities.to_color_label(pred_np))
cv2.waitKey(0)
pass