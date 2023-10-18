#Code "repurposed" from the Fordham Computational Neuroscience Lab and MIT CSAIL
#PURPOSE: Pull out middle layer responses from a resnet18 model pretrained on the places365 dataset and store them nice and neatly in a csv

from PIL import Image
import torchvision
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.transforms as transforms
import pandas as pd

transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])

#You can change the model, just make sure to use the proper .pth weight file as well.

arch = 'resnet18'

#http://places2.csail.mit.edu/models_places365/{MODEL_NAME}_places365.pth.tar for checkpoint, put your model name in the space: resnet18, alexnet, resnet50 etc

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load("C:\\Users\\gkrul\\OneDrive\\Desktop\\NN\P365\\resnet18_places365.pth.tar", map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
mod_lay_list=list(model.children())

responseMat=np.zeros((36500,512))

filler = np.zeros((1,512))
k=0

for root, dirs, files in os.walk("C:\\Users\\gkrul\\OneDrive\\Desktop\\NN\\val", topdown=False):
    
    for name in files:
        
        img=Image.open(os.path.join(root, name))
        img_t=transform(img)
        batch_t=torch.unsqueeze(img_t,0)
        if np.shape(batch_t)[1] < 3:
            responseMat[k,:]=filler
            k=k+1
            continue
        sub_net=nn.Sequential(*mod_lay_list[:8])
        out=sub_net(batch_t)
        outNP=out.detach().numpy()
        outVec=outNP.max(axis=2).max(axis=2)
        responseMat[k,:]=outVec
        k=k+1      

df=pd.DataFrame(responseMat)
df.to_csv('P365subnetResponse.csv')

df = pd.read_csv("C:\\Users\\gkrul\\OneDrive\\Desktop\\NN\\SubnetResponsesP365.csv")

names = []
gtlabels = []

for root, dirs, files in os.walk("C:\\Users\\gkrul\\OneDrive\\Desktop\\NN\\val", topdown=False):
    
    for name in files:
        
        names.append(os.path.basename(name))
        gtlabels.append(os.path.basename(root))
        
df.insert(loc = 0, column = 'col2', value = gtlabels)
df.insert(loc = 0, column = 'col1', value = names)

df.to_csv("C:\\Users\\gkrul\\OneDrive\\Desktop\\NN\\SubnetResponsesP365.csv", index=False)
