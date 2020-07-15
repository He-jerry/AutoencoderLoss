import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
from torchvision import  datasets
import cv2
from PIL import Image

img_transform = transforms.Compose([
    transforms.ToTensor()
])
 
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        nets=torchvision.models.vgg16(pretrained=True).features
        netlist=list(nets.children())
        vgglist=[]
        vgglist.append(netlist[0])
        vgglist.append(netlist[1])
        vgglist.append(netlist[5])
        vgglist.append(netlist[6])
        vgglist.append(netlist[10])
        vgglist.append(netlist[11])
        vgglist.append(netlist[17])
        vgglist.append(netlist[18])
        self.encoder = nn.Sequential(*vgglist)
        for p in self.parameters():
            p.requires_grad = False
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1),  # b, 1, 28, 28
            nn.ReLU(True)
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
 
 
model = autoencoder()
model=torch.load("/public/zebanghe2/utoencoder/conv_autoencoder40.pth")
model.cuda()
print(model)
model.eval()
import numpy as np

for i in range(0,256):
 img1=cv2.imread("/public/zebanghe2/utoencoder/9Son86aQI4M_4_lm.jpg")
 img2=cv2.imread("/public/zebanghe2/utoencoder/9Son86aQI4M_4_lt.jpg")
 img1[img1==0]=i
 img2[img2==0]=i
 img1=img_transform(img1)
 img2=img_transform(img2)
 img1=img1.cuda()
 img2=img2.cuda()
 img1=img1.unsqueeze(0)
 img2=img2.unsqueeze(0)
 criterion=torch.nn.L1Loss()

 with torch.no_grad():
  s1=model(img1)
  s2=model(img2)
 loss=criterion(s1,s2)
 print(loss.item())