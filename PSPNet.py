import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import cv2
from torchvision import transforms
import numpy as np
import utilities






class CustomResNet50WithDilation(nn.Module):
    def __init__(self):
        super(CustomResNet50WithDilation, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])

        self.layer2=resnet.layer2
        # self.layer2[0].conv2=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)
        # self.layer2[0].downsample[0]=nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # self.layer2[1].conv2=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)
        # self.layer2[2].conv2=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)
        # self.layer2[3].conv2=nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)

        self.layer3=resnet.layer3
        self.layer3[0].conv2=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)
        self.layer3[0].downsample[0]=nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer3[1].conv2=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)
        self.layer3[2].conv2=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)
        self.layer3[3].conv2=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)
        self.layer3[4].conv2=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)
        self.layer3[5].conv2=nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2,2), dilation=(2,2),bias=False)

        self.layer4=resnet.layer4
        self.layer4[0].conv2=nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4,4), dilation=(4,4),bias=False)
        self.layer4[0].downsample[0]=nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer4[1].conv2=nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4,4), dilation=(4,4),bias=False)
        self.layer4[2].conv2=nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4,4), dilation=(4,4),bias=False)

    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x

class PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in pool_sizes])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+len(pool_sizes)*512, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
    def _make_stage(self, in_channels, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=size),
            nn.Conv2d(in_channels, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        for stage in self.stages:
            tmp=stage(x)
            tmp=F.interpolate(tmp, size=(h, w), mode='bilinear', align_corners=False)
            pyramids.append(tmp)
        output = torch.cat(pyramids, dim=1)
        return self.bottleneck(output)

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        self.encoder = CustomResNet50WithDilation()
        self.decoder = PSPModule(in_channels=2048)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(num_classes*1024, 25, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.UpsamplingBilinear2d(scale_factor=8),
        )
        self.convs = nn.ModuleList([nn.Conv2d(2048, 1024, kernel_size=1) for _ in range(num_classes)])
    def forward(self, x):
        encoder_out=self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        class_specific_features = [conv(decoder_out) for conv in self.convs]
        output = torch.cat(class_specific_features, dim=1)
        output=self.segmentation_head(output)
        return output

encoder=CustomResNet50WithDilation()
decoder=PSPModule(2048)
model=PSPNet(25)
print(model)
X=torch.randn(6,3,512,512)
rst=model(X)
pass

def load_image_and_label(image_path,label_path,target_size):
    image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size,interpolation=cv2.INTER_LINEAR)
    label=cv2.imread(label_path,cv2.IMREAD_UNCHANGED)
    label = cv2.resize(label, target_size,interpolation=cv2.INTER_NEAREST)
    return image,label

# #device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Current used device is {device}")
# # model = models.segmentation.deeplabv3_resnet50(pretrained=True)
# # model.classifier[4] = nn.Conv2d(256, 25, kernel_size=(1, 1), stride=(1, 1))
# #model=NN.PSPNetWithDeepSupervision(25)
# # model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")
# # model.decode_head.classifier=nn.Conv2d(512, 25, kernel_size=(1, 1), stride=(1, 1))
# # model.auxiliary_head.classifier=nn.Conv2d(256, 25, kernel_size=(1, 1), stride=(1, 1))
# model=PSPNet(25)
# loaded_model = model#MyDeeplabV3Net(25,(6, 12, 18)) 
# loaded_model.load_state_dict(torch.load("C:/Users/SenGao/Downloads/model_250.pth"))
# loaded_model.to(device)
# loaded_model.eval()

# image_path="C:/Users/SenGao/Downloads/RUGD_ws/RUGD_frames-with-annotations/trail-3/trail-3_01961.png"
# label_path=image_path.replace("RUGD_frames-with-annotations", "RUGD_annotations_index")
# target_size=(512,512)
# class_num=25

# correct_pixel_sum=0
# pixel_sum=0
# total_intersection=np.zeros(class_num,dtype=np.int64)
# total_union=np.zeros(class_num,dtype=np.int64)
# with torch.no_grad():
#     image,label=load_image_and_label(image_path,label_path,target_size)
#     image_tensor=transforms.ToTensor()(image).unsqueeze(0)
#     output= loaded_model(image_tensor.to(device))
#     #prob = torch.softmax(output['out'], dim=1)
#     #prob = torch.softmax(output, dim=1)
#     pred = torch.argmax(output, dim=1)
#     pred_np = pred.cpu().numpy().reshape(512,512).astype(np.uint8)
#     correct_pixel_sum+=(pred_np==label).sum().item()
#     pixel_sum+=label.size

# pixel_accuracy=correct_pixel_sum/pixel_sum
# print(pixel_accuracy)
# cv2.imshow("image",image)
# cv2.imshow("GT",utilities.to_color_label(label))
# cv2.imshow("pred",utilities.to_color_label(pred_np))
# cv2.waitKey(0)
# pass

