from torch.utils.data import Dataset
import os
import pandas as pd
from dgl.data.utils import load_graphs
import dgl
import torch

# Set gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class custom_loader(Dataset):

    def __init__(self,path,type,pkpath=None):
        self.path = path
        self.type = type

        if pkpath == None:
            self.bind = False
        else:
            self.bind = True
            self.targets = pd.read_csv(pkpath)
            self.targets.set_index('PDB', inplace = True)

        if self.type == 'ligand' or self.type == 'both':
            self.ligs = [i[:-4] for i in os.listdir(path + '/ligand')]
        if self.type == 'ligand' or self.type == 'both':
            self.receptors = [i[:-4] for i in os.listdir(path + '/receptor')]
        
        if self.type == 'both':
            if len(self.receptors) != len(self.ligs):
                print('''Receptor and ligand directories don't match, terminating.''')
                exit()
        
    def __len__(self):
        if self.type == 'ligand' or 'both':
            return len(self.ligs)
        elif self.type == 'receptor':
            return len(self.receptors)

    def __getitem__(self,idx):

        if self.bind == True:
            if self.type == 'ligand' or self.type == 'both':
                pk = self.targets.loc[self.ligs[idx]].values[0]
            elif self.type == 'receptor':
                pk = self.targets.loc[self.receptors[idx]].values[0]

        if self.type == 'ligand':
            graph = load_graphs(self.path + '/ligand/' + self.ligs[idx] + '.bin')[0][0]
            if self.bind == True:
                return graph, pk
            else:
                return graph, self.ligs[idx]

        elif self.type == 'receptor':
            graph = load_graphs(self.path + '/receptor/' + self.receptors[idx] + '.bin')[0][0]
            if self.bind == True:
                return graph, pk
            else:
                return graph, self.receptors[idx]

        if self.type == 'both':
            liggraph = load_graphs(self.path + '/ligand/' + self.ligs[idx] + '.bin')[0][0]
            recgraph = load_graphs(self.path + '/receptor/' + self.receptors[idx] + '.bin')[0][0]
            if self.bind == True:
                return liggraph, recgraph, pk
            else:
                return liggraph, recgraph, self.ligs[idx]

# custom collate returns one or two batched graphs and a torch tensor of pK values if available
# code is 1 or 2 graphs then pk or no pk
def custom_collate_10(data):

    graphls = [i[0] for i in data]
    namels = [i[1] for i in data]
    batched_graph = dgl.batch(graphls)
    batched_graph = batched_graph.to(device)
    return batched_graph, namesls

def custom_collate_11(data):
    graphls = [i[0] for i in data]
    batched_graph = dgl.batch(graphls)
    pk = [i[1] for i in data]
    pk = torch.Tensor(pk)
    batched_graph = batched_graph.to(device)
    pk = pk.to(device)
    return batched_graph, pk

def custom_collate_20(data):
    liggraphls = [i[0] for i in data]
    lig_batched_graph = dgl.batch(liggraphls)
    recgraphls = [i[1] for i in data]
    rec_batched_graph = dgl.batch(recgraphls)
    namels = [i[2] for i in data]
    lig_batched_graph = lig_batched_graph.to(device)
    rec_batched_graph = rec_batched_graph.to(device)
    return lig_batched_graph, rec_batched_graph, namels

def custom_collate_21(data):
    liggraphls = [i[0] for i in data]
    lig_batched_graph = dgl.batch(liggraphls)
    recgraphls = [i[1] for i in data]
    rec_batched_graph = dgl.batch(recgraphls)
    pk = [i[2] for i in data]
    pk = torch.Tensor(pk)
    lig_batched_graph = lig_batched_graph.to(device)
    rec_batched_graph = rec_batched_graph.to(device)
    pk = pk.to(device)
    return lig_batched_graph, rec_batched_graph, pk