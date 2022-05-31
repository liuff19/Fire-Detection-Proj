# Author : Fangfu & Zerun
# Univ : Tsinghua Univ, Dept of EE
# Here we use resnet to relize fire detection

from __future__ import print_function, division
from sympy import false

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.nn import init
from tqdm import tqdm
from run_helpers import *
from mae_classifier_model import *
# we can add more transforms here if we need more deformable data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # adapt to the imageNet_size
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # same as above
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
}

def train_model(model, criterion, optimizer, scheduler, num_epochs, ckpt_path):
    print('-------------------------training------------------------------')
    since = time.time()
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # each epoch has 2 phases: train and validation
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # here we iter the data
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # especially in MAE
                inputs = mae_net(inputs,mask_ratio=0)
                inputs = inputs[:,0,:].squeeze()
                outputs = model(inputs)

                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # bp operation is only done here
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # we compute the loss and acc here
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深度复制mo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save best model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    ckpt_name = ckpt_path
    torch.save({'state_dict': best_model_wts}, ckpt_name)
    print(f'[Info] Model saved in {ckpt_name} \n')

def test_model(model, ckpt_path):
    print('------------------------------------test-------------------------------')
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print('[Info] Load model from {}'.format(ckpt_path))

    model.eval()
    test_corrects = 0.0
    TP = 0.0
    FP =0.0
    T = 119
    F = 107
    for inputs, labels in tqdm(dataloaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        inputs = mae_net(inputs,mask_ratio=0)
        inputs = inputs[:,0,:].squeeze()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)
        for (pred, label) in zip(preds, labels):
            if pred==0:
                if label==0:
                    TP+=1
                elif label==1:
                    FP+=1
    acc = test_corrects.double() / dataset_sizes['test']
    print(f'[Info] Test accuracy = {acc}')
    print(f'[Info] True positive rate = {TP/T}')
    print(f'[Info] False positive rate = {FP/F}')

def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=1)

if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    parser = config_parser()
    args = parser.parse_args()

    device = torch.device(args.device)

    # define Network
    model_ft = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.Tanh(),
        torch.nn.Linear(512,2)
    )
    model_ft.to(args.device)
    mae_net = classifier_mae()
    mae_net = mae_net.to(args.device)

    data_dir = args.data_path
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True if x=='train' else false, num_workers=0)
              for x in ['train', 'val','test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes


    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    if args.backbone_print:
        print(model_ft)
    # -- run the code for training and validation
    if args.mode == 'train':
        train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=args.epoch, ckpt_path=args.ckpt_path)

    if args.mode == 'test':
        args.ckpt_path = '/opt/data2/lff/Fire Detection/saved_models/FireMAE.pth'
        test_model(model_ft, args.ckpt_path)
    
    
