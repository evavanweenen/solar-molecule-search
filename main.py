# Task 4: smiles
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np 
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA

from data import SmilesDataset
from model import FeedforwardNN
from train import optimize

DATADIR = './data/'
MODELNAME = 'feedforwardnn'
LEARNING_RATE = 1e-4
PATIENCE = 10
SEED = 42
EPOCHS = 600

# --------------- pretraining
data_train = SmilesDataset(f'{DATADIR}pretrain', split='train', random_state=SEED, test_size=0.1)
data_val = SmilesDataset(f'{DATADIR}pretrain', split='val', random_state=SEED, test_size=0.1)

p = PCA(n_components=228)
data_train.X = torch.tensor(p.fit_transform(data_train.X)).float()
data_val.X = torch.tensor(p.transform(data_val.X)).float()

trainloader = DataLoader(data_train, batch_size=1000, shuffle=True)
valloader = DataLoader(data_val, batch_size=1000, shuffle=True)

model = FeedforwardNN()
criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(model)
model.to(device)

optimize(model, criterion, optimizer, scheduler, trainloader, valloader, data_train, data_val, 
    device, epochs=EPOCHS, patience=PATIENCE, mname=MODELNAME)

# --------------- finetuning (1)
data_train = SmilesDataset(f'{DATADIR}train', split='train', random_state=SEED, test_size=0.1)
data_val = SmilesDataset(f'{DATADIR}train', split='val', random_state=SEED, test_size=0.1)

data_train.X = torch.tensor(p.transform(data_train.X)).float()
data_val.X = torch.tensor(p.transform(data_val.X)).float()

trainloader = DataLoader(data_train, batch_size=10, shuffle=True)
valloader = DataLoader(data_val, batch_size=10, shuffle=True)

# load pre-trained model
model = torch.load(MODELNAME)

# freeze first layers
for layer in model.module.model[:-1]:
    for param in layer.parameters():
        param.requires_grad = False

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE*100)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimize(model, criterion, optimizer, scheduler, trainloader, valloader, data_train, data_val,
    device, epochs=EPOCHS, patience=PATIENCE, mname=MODELNAME+'_finetune')

# --------------- finetuning (2)
trainloader = DataLoader(data_train, batch_size=10, shuffle=True)
valloader = DataLoader(data_val, batch_size=10, shuffle=True)

# load pre-trained model
model = torch.load(MODELNAME+'_finetune')

# unfreeze first layers
for layer in model.module.model[:-1]:
    for param in layer.parameters():
        param.requires_grad = True

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)#

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimize(model, criterion, optimizer, scheduler, trainloader, valloader, data_train, data_val,
    device, epochs=20, patience=PATIENCE, mname=MODELNAME+'_finetune2')

# --------------- evaluate
data_test = SmilesDataset(f'{DATADIR}test', split='test', recalc_smiles=True)

data_test.X = torch.tensor(p.transform(data_test.X)).float()

model = torch.load(MODELNAME+'_finetune2')
model.to(device)
model.eval()

with torch.no_grad():
    submission = model(data_test.X)

submission = pd.DataFrame(submission.cpu().numpy(), index=data_test.S.index, columns=['y'])
submission.to_csv('submission.txt')