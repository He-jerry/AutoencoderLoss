#! /usr/bin/env python

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
 
num_epochs = 100
batch_size = 128
learning_rate = 1e-4
train_data = datasets.ImageFolder(r'/public/zebanghe2/utoencoder/',transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
]))
train_loader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True)
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
 
 
model = autoencoder().cuda()
print(model)
criterion = nn.MSELoss()
criterion2 = nn.L1Loss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
                             weight_decay=1e-5)
 
for epoch in range(100):
    for data in train_loader:
        img, _ = data
        img = Variable(img).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        loss2=criterion2(output,img)
        # ===================backward====================
        optimizer.zero_grad()
        losstt=loss+loss2
        losstt.backward()
        optimizer.step()
        print(losstt.item())
    # ===================log========================
    if(epoch%10==0):
      torch.save(model, 'conv_autoencoder'+str(epoch)+'.pth')
 
