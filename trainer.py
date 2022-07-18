import models.score_model as score_model

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
p.add_argument('--train', type=str, default='hidden_layers/receptor/train', help='path to train binary layers')
p.add_argument('--val', type=str, default='hidden_layers/receptor/val', help='path to val binary layers')
p.add_argument('--test', type=str, default='hidden_layers/receptor/test', help='path to test binary layers')
p.add_argument('--model_output', type=str, default='runs/score/ligand_trained.pt', help='path to .pt file for saving model')
args = p.parse_args()

# Load targets
targets = pd.read_csv('bindingdata.csv')
targets.set_index('PDB', inplace = True)

# Load validation data
valgraphls = []
valnames = []

for file in os.listdir(args.val):
    tempvalgraphls, labeldict = load_graphs(args.val+'/'+file)
    tempvalnames = list(labeldict.keys())
    valnames = valnames + tempvalnames
    labels = [int(i) for i in labeldict.values()]
    smallest = min(labels)
    labels = [i-smallest for i in labels]
    tempvalgraphls = [tempvalgraphls[labels[i]] for i in range(len(tempvalgraphls))]
    valgraphls = valgraphls + tempvalgraphls

val_batched_graph = dgl.batch(valgraphls)
valpK =  targets.loc[valnames].values.flatten()
valpK = torch.Tensor(valpK)

# Create the model with given dimensions
model = score_model.GAT()
#model.to('cuda:0')

# Define loss
loss = nn.MSELoss()

# For optimisation
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# Training params
num_epochs = 30

for i in range(num_epochs):

    # Loop through each batch
    count=0
    for file in os.listdir(args.train):

        # Load training batch
        traingraphls, labeldict = load_graphs(args.train+'/'+file)
        labels = [int(i) for i in labeldict.values()]
        smallest = min(labels)
        labels = [i-smallest for i in labels]
        traingraphls = [traingraphls[labels[i]] for i in range(len(traingraphls))]
        trainnames = list(labeldict.keys())
        train_batched_graph = dgl.batch(traingraphls)
        trainpK =  targets.loc[trainnames].values.flatten()
        trainpK = torch.Tensor(trainpK)

        # Training loop

        pred = model(train_batched_graph)
        try:
            pred = model(train_batched_graph)
            optimizer.zero_grad()
            target = loss(pred,trainpK)
            target.backward()
            optimizer.step()
        except:
            continue

        print('Batch ' + str(count+1)+ ' training loss: ' + str(float(target)))

        count += 1

    # Validation
    with torch.no_grad():
        valpred = model(val_batched_graph)
        valloss = loss(valpred,valpK)
        print('\nIteration ' + str(i+1)+ ' validation loss: ' + str(float(valloss)) +'\n')
    

print('Training finished, saving model')
torch.save(model.state_dict(), args.model_output)

# Evaluation

# Load test data
testgraphls = []
testnames = []

for file in os.listdir(args.val):
    temptestgraphls, labeldict = load_graphs(args.test+'/'+file)
    temptestnames = list(labeldict.keys())
    testnames = testnames + temptestnames
    testgraphls = testgraphls + temptestgraphls
    labels = [int(i) for i in labeldict.values()]
    smallest = min(labels)
    labels = [i-smallest for i in labels]
    testgraphls = [testgraphls[labels[i]] for i in range(len(testgraphls))]
    break

test_batched_graph = dgl.batch(testgraphls)
testpK =  targets.loc[testnames].values.flatten()
testpK = torch.Tensor(testpK)

# Test prediction and loss
testpred = torch.squeeze(model(test_batched_graph, test_batched_graph.ndata['final_hidden'].float()))
testloss = loss(testpred,testpK)

print('\nTest loss: ' + testloss)
print('\nTest RMSE: ' + torch.sqrt(testloss))