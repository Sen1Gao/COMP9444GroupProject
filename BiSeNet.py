import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.dropout1=nn.Dropout2d(0.1)
        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)
    def forward(self, x):
        x=self.conv1(x)
        x=self.dropout1(x)
        x=self.conv2(x)
        x=self.dropout2(x)
        x=self.conv3(x)
        x=self.dropout3(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg=self.global_pool(x)
        avg=self.conv(avg)
        avg=self.bn(avg)
        attention = self.sigmoid(avg)
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x * attention


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.arm8x = AttentionRefinementModule(128,512)
        self.arm16x = AttentionRefinementModule(256,256)
        self.conv1=ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2=ConvBlock(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dropout1=nn.Dropout2d(0.1)
        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)
        
    def forward(self, x):
        x2 = self.resnet18.conv1(x)
        x2 = self.resnet18.bn1(x2)
        x2 = self.resnet18.relu(x2)
        x4 = self.resnet18.maxpool(x2)
        x4 = self.resnet18.layer1(x4)
        x8 = self.resnet18.layer2(x4)  
        x16 = self.resnet18.layer3(x8)
        
        avg=self.avg_pool(x16)
        avg_up=F.interpolate(avg, size=x16.size()[2:], mode='bilinear', align_corners=True)
        
        x16_arm=self.arm16x(x16)
        x16_arm=x16_arm+avg_up
        x16_arm_up=F.interpolate(x16_arm, size=x8.size()[2:], mode='bilinear', align_corners=True)
        x16_arm_up=self.conv1(x16_arm_up)
        x16_arm_up=self.dropout2(x16_arm_up)
        
        x8_arm=self.arm8x(x8)
        x8_arm=x8_arm+x16_arm_up
        x8_arm=self.conv2(x8_arm)
        x8_arm=self.dropout3(x8_arm)
        
        return x8_arm,x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, 1, 1, 0)
        self.dropout=nn.Dropout2d(0.4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, sp, cp):
        fusion = torch.cat([sp, cp], dim=1)
        fusion = self.conv(fusion)
        fusion=self.dropout(fusion)
        attention = self.attention(fusion)
        return fusion + fusion * attention


class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        self.feature_fusion = FeatureFusionModule(512,128)
        self.final_conv=nn.Conv2d(128, num_classes, 1,1,0)
        
    def forward(self, x):
        sp = self.spatial_path(x)
        cp,x = self.context_path(x)
        out = self.feature_fusion(sp,cp)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        return out