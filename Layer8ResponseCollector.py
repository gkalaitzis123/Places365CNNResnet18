#Code "repurposed" from the Fordham Computational Neuroscience Lab and MIT CSAIL
#PURPOSE: Pull out middle layer responses from a resnet18 model pretrained on the places365 dataset and store them nice and neatly in a csv

from PIL import Image
import pandas as pd
import torchvision
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.transforms as transforms

transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()])

#You can change the model, just make sure to use the proper .pth weight file as well.

arch = 'resnet18'

#http://places2.csail.mit.edu/models_places365/{MODEL_NAME}_places365.pth.tar for checkpoint, put your model name in the space: resnet18, alexnet, resnet50 etc

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load("yourpthfilehere", map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
mod_lay_list=list(model.children())

maxVec=[]
responseMat=np.zeros((5000,512))

for i in range(1):
    
    #replace imPath and the path in files with your own chosen path
    
    imPath="yourimgdirectiorypathhere"
    files=os.listdir("yourimgdirectiorypathhere")
    k=0
    fileList=[]
    for file in files:
        img=Image.open(imPath+'/'+file)
        img_t=transform(img)
        batch_t=torch.unsqueeze(img_t,0)
        sub_net=nn.Sequential(*mod_lay_list[:8])
        out=sub_net(batch_t)
        outNP=out.detach().numpy()
        outVec=outNP.max(axis=2).max(axis=2)
        maxVec.append(out.argmax())
        responseMat[k,:]=outVec
        fileList.append(file)
        k=k+1

responseMat=responseMat[:(k),:]
df=pd.DataFrame(responseMat)
df.index=fileList
df.to_csv('subnetResponses.csv')
