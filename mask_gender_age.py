import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from time import time

import torch
import torch.utils.data as data
import albumentations
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import torchvision.models as models
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from dataset import *
test_dir = '/opt/ml/input/data/train'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device:[%s]."%(device))

test_dir=os.path.join(test_dir,'images')
from loss import *

mean=(0.5, 0.5, 0.5)
std=(0.2, 0.2, 0.2)
transform = get_transforms(mean=mean, std=std)

dataset = MaskBaseDataset(
    img_dir=test_dir
)

# train dataset과 validation dataset을 8:2 비율로 나눕니다.
n_val = int(len(dataset) * 0.2)
n_train = len(dataset) - n_val
train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

# 각 dataset에 augmentation 함수를 설정합니다.
train_dataset.dataset.set_transform(transform['train'])
val_dataset.dataset.set_transform(transform['val'])

#train, val에 대해서 dataloader생성. batchsize를 바꾸면 승내는 loss function이 좀 있어서 batch가 강제되거나 loss 를 CE만 썼다.

train_loader = data.DataLoader(
    train_dataset,
    batch_size=12,
    num_workers=4,
    shuffle=True
)

val_loader = data.DataLoader(
    val_dataset,
    batch_size=12,
    num_workers=4,
    shuffle=False
)
criterion = nn.CrossEntropyLoss()

# MNIST에서 긁어온 채점 프로그램 구조. 여기서 나름 변수 통일시킨다고 그놈의 to(device)라던가 클라스를 좀 배웠다. 이거 아니였으면 GPU사용이라던가 모델의 작동 구도를 모르고 프로젝트가 넘어갔을거 같다.

def func_eval(model,data_iter,device):
    with torch.no_grad():
        model.eval() # evaluate (affects DropOut and BN)
        n_total,n_correct = 0,0
        for i, data in enumerate(train_loader,0):
            inputs,labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs=net(inputs)

            _,y_pred = torch.max(outputs.data,1)
            n_correct += (
                y_pred==outputs
            ).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr
print ("Done")

# 학습모델 call


from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b4',num_classes=18)
# fc 제외하고 freeze
for n, p in model.named_parameters():
    if '_fc' not in n:
        p.requires_grad = False

# pret_res50=models.resnet50(pretrained=True)
# for para in pret_res50.parameters():
#     para.requires_grad=False
# pret_res50.fc.weight.requires_grad = True
# pret_res50.fc=nn.Linear(in_features=2048, out_features=18, bias=True)


# net=models.vgg16(num_classes=18)

#작고 소듕한 학습. While문을 통한 funnel을 시도해 보았지만 epoch 증가정도의 효과만 가져왔기에 효과는 미미했다.

lr = 1e-4
# lambda1 = lambda epoch: epoch // 30 - 솔직히 긁어온 후보군 중 하나인데 논리가 너무 안보여서 그냥 뺏다
lambda2 = lambda epoch: 0.95 ** epoch
optimizer = AdamP(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = LambdaLR(optimizer, lr_lambda=[lambda2])
model=model.to(device)
# 이부분도 baseline과 MNIST 예제에서 긁어온게 많다. 그래도 요 구조는 한동안 계속 우려먹지 싶다
for epoch in range(50):
    running_loss =0.0
    for i, data in enumerate(train_loader,0):
        inputs,labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs=model. forward(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if i%200==199:
            print('[%d %5d] loss : %.3f'%(epoch+1,i+1,running_loss/200))
            running_loss=0.0

print("finish")

EFF_adamp_scheduler=model
PATH= '/opt/ml/EFF_adamp_scheduler.pth'
torch.save(EFF_adamp_scheduler.state_dict(), PATH)