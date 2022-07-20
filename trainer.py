import models.score_model as score_model
from commons.custom_data_loader import custom_loader, custom_collate_10, custom_collate_20, custom_collate_11, custom_collate_21
from torch.utils.data import DataLoader
from copy import deepcopy
import os
import yaml
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
def parse_arguments(arglist = None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/trainer.yml')
    p.add_argument('--hidden_layers', type=str, default='hidden_layers', help='path to hidden binary layers')
    p.add_argument('--type', type=str, default='ligand', help='ligand, receptor, or both')
    p.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    p.add_argument('--model_output', type=str, default='runs/score/ligand_trained.pt', help='path to .pt file for saving model')

    cmdline_parser = deepcopy(p)
    args = p.parse_args(arglist)
    clear_defaults = {key: argparse.SUPPRESS for key in args.__dict__}
    cmdline_parser.set_defaults(**clear_defaults)
    cmdline_parser._defaults = {}
    cmdline_args = cmdline_parser.parse_args(arglist)
        
    return args, cmdline_args

args, cmdline_args = parse_arguments()
if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            if key in cmdline_args:
                continue
            arg_dict[key] = value
    args.config = args.config.name

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
            pred = model(train_batched_graph)
            optimizer.zero_grad()
            target = loss(pred,trainpK)
            target.backward()
            optimizer.step()
            print('Batch ' + str(idx+1)+ ' training loss: ' + str(float(target)))

        # Validation
        with torch.no_grad():
            for val_batched_graph, valpK in valloader:
                valpred = model(val_batched_graph)
                valloss = loss(valpred,valpK)
                print('\nIteration ' + str(i+1)+ ' validation loss: ' + str(float(valloss)) +'\n')

    else:
        for idx, (lig_batched_graph, rec_batched_graph, trainpK) in enumerate(trainloader):

            # Training loop
            pred = model(lig_batched_graph, rec_batched_graph)
            optimizer.zero_grad()
            target = loss(pred,trainpK)
            target.backward()
            optimizer.step()
            print('Batch ' + str(idx+1)+ ' training loss: ' + str(float(target)))      

        # Validation
        with torch.no_grad():
            for val_lig_batched_graph, val_rec_batched_graph, valpK in valloader:
                valpred = model(val_lig_batched_graph, val_rec_batched_graph)
                valloss = loss(valpred,valpK)
                print('\nIteration ' + str(i+1)+ ' validation loss: ' + str(float(valloss)) +'\n')
    

print('Training finished, saving model')
torch.save(model.state_dict(), args.model_output)

# Evaluation
with torch.no_grad():
    if model1:
        for test_batched_graph, testpK in testloader:
            testpred = model(test_batched_graph)
            testloss = loss(testpred,testpK)
    else:
        for test_lig_batched_graph, test_val_batched_graph, testpK in testloader:
            testpred = model(test_lig_batched_graph, test_val_batched_graph)
            testloss = loss(testpred,testpK)

print('\nTest loss: ' + testloss)
print('\nTest RMSE: ' + torch.sqrt(testloss))