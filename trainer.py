import models.score_model as score_model

import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dgl.data.utils import load_graphs

# Load training data
traingraphls, labeldict = load_graphs('hidden_layers/test.bin')
trainnames = list(labeldict.keys())

# Load validation data
valgraphls, labeldict = load_graphs('hidden_layers/test.bin')
valnames = list(labeldict.keys())

# Create the model with given dimensions
model = score_model.GCN(64, 16, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

# For early stopping
stop = False
patience = 5
counter = 0
prevvalloss = torch.tensor(float('inf'))

while not stop:

    # Training loop
    pred = torch.squeeze(model(train_batched_graph, train_batched_graph.ndata['final_hidden'].float()))
    target = loss(pred,trainpK)
    optimizer.zero_grad()
    target.backward(retain_graph=True)
    optimizer.step()

    # Validation
    with torch.no_grad():

        valpred = torch.squeeze(model(val_batched_graph, val_batched_graph.ndata['final_hidden'].float()))
        valloss = loss(pred,valpK)
        print('Validation loss: ' + str(float(valloss)))

        # Check if validation loss is increasing
        if valloss > prevvalloss:
            counter += 1
        else:
            counter = 0
        
        if counter == patience:
            print('Validation loss increasing, saving model')
            stop = True
            torch.save(model.state_dict(), 'runs/scoremodel/bestcheckpoint.pt')

print('Training finished')