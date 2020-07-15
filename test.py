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
 
num_epochs = 100
batch_size = 128
learning_rate = 1e-4
train_data = datasets.ImageFolder(r"/public/zebanghe2/utoencoder/",transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
]))
train_loader = torch.utils.data.DataLoader(train_data,batch_size=1,shuffle=False)
print(len(train_loader))

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
model=torch.load("/public/zebanghe2/utoencoder/conv_autoencoder90.pth")
model.cuda()
print(model)
model.eval()
import numpy as np


def denormalize(image):
  #image=transforms.ToTensor()(image)
  #image=transforms.ToPILImage()(image)
  #image = transforms.Normalize(-mean/std,1/std)(image) #denormalize
  image = image.permute(1,2,0) #Changing from 3x224x224 to 224x224x3
  #image = torch.clamp(image,0,1)
  return image

count=0
for data in train_loader:
 count=count+1
 img, _ = data
 img = Variable(img).cuda()
 with torch.no_grad():
   output = model(img)
 #print(output)
 out=denormalize(output.cpu().data[0])
 print(out.shape)
 out=out.numpy()
 out=out*255
 cv2.imwrite("/public/zebanghe2/utoencoder/output/"+str(count)+' '+str(train_data[count][1])+'.jpg',out)

 