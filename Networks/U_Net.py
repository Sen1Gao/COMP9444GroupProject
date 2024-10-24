"""
Group: Team1024
File name: U-Net.py
Author: Sen Gao
"""

import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self,class_num):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = CBR(3, 32)
        self.encoder2 = CBR(32, 64)
        self.dropout2=nn.Dropout2d(0.1)
        self.encoder3 = CBR(64, 128)
        self.dropout3=nn.Dropout2d(0.2)
        self.encoder4 = CBR(128, 256)
        self.dropout4=nn.Dropout2d(0.3)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(256, 512)
        self.dropout5=nn.Dropout2d(0.5)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = CBR(512, 256)
        self.dropout6=nn.Dropout2d(0.3)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = CBR(256, 128)
        self.dropout7=nn.Dropout2d(0.2)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = CBR(128, 64)
        self.dropout8=nn.Dropout2d(0.1)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = CBR(64, 32)

        self.conv_last = nn.Conv2d(32, class_num, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.dropout2(self.encoder2(self.pool(enc1)))
        enc3 = self.dropout3(self.encoder3(self.pool(enc2)))
        enc4 = self.dropout4(self.encoder4(self.pool(enc3)))

        bottleneck = self.dropout5(self.bottleneck(self.pool(enc4)))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dropout6(self.decoder4(dec4))

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dropout7(self.decoder3(dec3))

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dropout8(self.decoder2(dec2))

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_last(dec1)