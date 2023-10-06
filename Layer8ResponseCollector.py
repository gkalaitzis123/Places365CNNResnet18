{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0b5a5207-4d3b-4402-8c3a-9297659a4ce1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[tensor(18490), tensor(18391), tensor(18483), tensor(18499), tensor(23138), tensor(18386), tensor(14289), tensor(3594), tensor(18385), tensor(18393), tensor(14283), tensor(14284), tensor(2564), tensor(3595), tensor(14276), tensor(15998), tensor(18398), tensor(18392), tensor(14269), tensor(10622), tensor(1249), tensor(15992), tensor(18049), tensor(24181), tensor(24181), tensor(24186), tensor(7374), tensor(23199), tensor(5897), tensor(23203)]\n",
      "[[0.22748324 0.0655518  2.2033577  ... 0.6440289  0.32371482 0.25328803]\n",
      " [0.69125402 0.         0.82537472 ... 0.15920612 3.24844098 0.32449418]\n",
      " [1.09887981 0.10840626 1.19625342 ... 0.14328259 0.72366571 0.        ]\n",
      " ...\n",
      " [0.25914183 0.         3.73374724 ... 1.59402239 3.81237841 0.02676585]\n",
      " [0.         0.02936055 1.70935822 ... 0.         2.74975944 0.98920327]\n",
      " [0.81100059 1.79920995 1.72326398 ... 0.12309504 2.95580435 0.42523643]]\n"
     ]
    }
   ],
   "source": [
    "#Code \"repurposed\" from the Fordham Computational Neuroscience Lab and MIT CSAIL\n",
    "#PURPOSE: Pull out middle layer responses from a resnet18 model pretrained on the places365 dataset and store them nice and neatly in a csv\n",
    "\n",
    "from PIL import Image\n",
    "import pandas as pd\n",
    "import torchvision\n",
    "from torchvision import models\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import numpy as np\n",
    "import os\n",
    "import torchvision.transforms as transforms\n",
    "\n",
    "transform=transforms.Compose([\n",
    "    transforms.Resize(256),\n",
    "    transforms.CenterCrop(224),\n",
    "    transforms.ToTensor()])\n",
    "\n",
    "#You can change the model, just make sure to use the proper .pth weight file as well.\n",
    "\n",
    "arch = 'resnet18'\n",
    "\n",
    "#http://places2.csail.mit.edu/models_places365/{MODEL_NAME}_places365.pth.tar for checkpoint, put your model name in the space: resnet18, alexnet, resnet50 etc\n",
    "\n",
    "model = models.__dict__[arch](num_classes=365)\n",
    "checkpoint = torch.load(\"yourpthfilehere\", map_location=lambda storage, loc: storage)\n",
    "state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}\n",
    "model.load_state_dict(state_dict)\n",
    "model.eval()\n",
    "mod_lay_list=list(model.children())\n",
    "\n",
    "maxVec=[]\n",
    "responseMat=np.zeros((5000,512))\n",
    "\n",
    "for i in range(1):\n",
    "    \n",
    "    #replace imPath and the path in files with your own chosen path\n",
    "    \n",
    "    imPath=\"yourimgdirectiorypathhere\"\n",
    "    files=os.listdir(\"yourimgdirectiorypathhere\")\n",
    "    k=0\n",
    "    fileList=[]\n",
    "    for file in files:\n",
    "        img=Image.open(imPath+'/'+file)\n",
    "        img_t=transform(img)\n",
    "        batch_t=torch.unsqueeze(img_t,0)\n",
    "        sub_net=nn.Sequential(*mod_lay_list[:8])\n",
    "        out=sub_net(batch_t)\n",
    "        outNP=out.detach().numpy()\n",
    "        outVec=outNP.max(axis=2).max(axis=2)\n",
    "        maxVec.append(out.argmax())\n",
    "        responseMat[k,:]=outVec\n",
    "        fileList.append(file)\n",
    "        k=k+1\n",
    "\n",
    "responseMat=responseMat[:(k),:]\n",
    "df=pd.DataFrame(responseMat)\n",
    "df.index=fileList\n",
    "df.to_csv('subnetResponses.csv')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
