import models.score_model as score_model
from commons.custom_data_loader import custom_loader, custom_collate_10, custom_collate_20, custom_collate_11, custom_collate_21
from torch.utils.data import DataLoader

import os
import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from dgl.data.utils import load_graphs

import argparse

# Parse arguments
p = argparse.ArgumentParser()
p.add_argument('--hidden_layers', type=str, default='hidden_layers', help='path to hidden binary layers')
p.add_argument('--type', type=str, default='ligand', help='ligand, receptor, or both')
p.add_argument('--batch_size', type=int, default=100, help='batch size for training')
p.add_argument('--model_output', type=str, default='runs/score/ligand_trained.pt', help='path to .pt file for saving model')
args = p.parse_args()

# model 1 - true, model 2 - false
if args.type == 'ligand' or args.type == 'receptor':
    model1 = True
else:
    model1 = False

# Set gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the data
traindata = custom_loader(args.hidden_layers + '/train',args.type,'bindingdata.csv')
valdata = custom_loader(args.hidden_layers + '/val',args.type,'bindingdata.csv')
testdata = custom_loader(args.hidden_layers + '/test',args.type,'bindingdata.csv')

if model1:
    trainloader = DataLoader(traindata,shuffle=True,batch_size=args.batch_size,collate_fn=custom_collate_11)
    valloader = DataLoader(valdata,shuffle=False,batch_size=len(valdata),collate_fn=custom_collate_11)
    testloader = DataLoader(testdata,shuffle=False,batch_size=len(testdata),collate_fn=custom_collate_11)

else:
    trainloader = DataLoader(traindata,shuffle=True,batch_size=args.batch_size,collate_fn=custom_collate_21)
    valloader = DataLoader(valdata,shuffle=False,batch_size=len(valdata),collate_fn=custom_collate_21)
    testloader = DataLoader(testdata,shuffle=False,batch_size=len(testdata),collate_fn=custom_collate_21)

# Load the model
if model1:
    model = score_model.GAT1()
else:
    model = score_model.GAT2()
model.to(device)

# Define loss
loss = nn.MSELoss()

# For optimisation
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training params
num_epochs = 30

for i in range(num_epochs):

    if model1:
        for idx, (train_batched_graph, trainpK) in enumerate(trainloader):

            # Training loop
            try:
                pred = model(train_batched_graph)
                optimizer.zero_grad()
                target = loss(pred,trainpK)
                target.backward()
                optimizer.step()
            except:
                continue

            print('Batch ' + str(idx+1)+ ' training loss: ' + str(float(target)))

            count += 1

        # Validation
        with torch.no_grad():
            val_batched_graph, valpK = valloader[0]
            valpred = model(val_batched_graph)
            valloss = loss(valpred,valpK)
            print('\nIteration ' + str(i+1)+ ' validation loss: ' + str(float(valloss)) +'\n')

    else:
        for idx, (lig_batched_graph, rec_batched_graph, trainpK) in enumerate(trainloader):

            # Training loop
            try:
                pred = model(lig_batched_graph, rec_batched_graph)
                optimizer.zero_grad()
                target = loss(pred,trainpK)
                target.backward()
                optimizer.step()
            except:
                continue

            print('Batch ' + str(idx+1)+ ' training loss: ' + str(float(target)))

            count += 1

        # Validation
        with torch.no_grad():
            val_lig_batched_graph, val_rec_batched_graph, valpK = valloader[0]
            valpred = model(val_lig_batched_graph, val_rec_batched_graph)
            valloss = loss(valpred,valpK)
            print('\nIteration ' + str(i+1)+ ' validation loss: ' + str(float(valloss)) +'\n')
    

print('Training finished, saving model')
torch.save(model.state_dict(), args.model_output)

# Evaluation
with torch.no_grad():
    if model1:
        test_batched_graph, testpK = testloader[0]
        testpred = model(test_batched_graph)
        testloss = loss(testpred,testpK)
    else:
        test_lig_batched_graph, test_val_batched_graph, testpK = testloader[0]
        testpred = model(test_lig_batched_graph, test_val_batched_graph)
        testloss = loss(testpred,testpK)

print('\nTest loss: ' + testloss)
print('\nTest RMSE: ' + torch.sqrt(testloss))