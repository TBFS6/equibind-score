import models.score_model as score_model

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
p.add_argument('--train', type=str, default='hidden_layers/train2.bin', help='path to train binary layers')
p.add_argument('--val', type=str, default='hidden_layers/val2.bin', help='path to val binary layers')
p.add_argument('--test', type=str, default='hidden_layers/test2.bin', help='path to test binary layers')
p.add_argument('--model_output', type=str, default='runs/score/best_checkpoint.pt', help='path to .pt file for saving model')
args = p.parse_args()

# Load training data
traingraphls, labeldict = load_graphs(args.train)
trainnames = list(labeldict.keys())

# Load validation data
valgraphls, labeldict = load_graphs(args.val)
valnames = list(labeldict.keys())

# Create the model with given dimensions
model = score_model.GAT()

# Load targets
targets = pd.read_csv('BindingData.csv')
del targets['SET']
targets.set_index('PDB', inplace = True)

# Remove missing ligands
namesbool = [(i in targets.index) for i in trainnames]
trainnames = [trainnames[i] for i in range(len(trainnames)) if namesbool[i]]
traingraphls = [traingraphls[i] for i in range(len(traingraphls)) if namesbool[i]]

namesbool = [(i in targets.index) for i in valnames]
valnames = [valnames[i] for i in range(len(valnames)) if namesbool[i]]
valgraphls = [valgraphls[i] for i in range(len(valgraphls)) if namesbool[i]]

# Batch graphs
train_batched_graph = dgl.batch(traingraphls)
val_batched_graph = dgl.batch(valgraphls)

# Labels for loss function
trainpK =  targets.loc[trainnames].values.flatten()
trainpK = torch.Tensor(trainpK)

# Labels for validation
valpK =  targets.loc[valnames].values.flatten()
valpK = torch.Tensor(valpK)

# Define loss
loss = nn.MSELoss()

# For optimisation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer,patience=0,factor=0.1)

# Training params
num_epochs = 30

for i in range(num_epochs):

    # Training loop
    pred = torch.squeeze(model(train_batched_graph, train_batched_graph.ndata['final_hidden'].float()))
    target = loss(pred,trainpK)
    optimizer.zero_grad()
    target.backward(retain_graph=True)
    optimizer.step()

    # Validation
    with torch.no_grad():
        valpred = torch.squeeze(model(val_batched_graph, val_batched_graph.ndata['final_hidden'].float()))
        valloss = loss(valpred,valpK)
        print('Iteration ' +str(i)+ ' validation loss: ' + str(float(valloss)))
    
    # Scheduler
    scheduler.step(valloss)

print('Training finished, saving model')
torch.save(model.state_dict(), args.model_output)

