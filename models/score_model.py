import dgl
from dgl.nn import GATConv
from dgl.nn import MaxPooling
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.gat1 = GATConv(64, 64, 10, allow_zero_in_degree=True)
        self.gat2 = GATConv(640,128,1, allow_zero_in_degree=True)
        self.maxpool = MaxPooling()
        self.lin1 = nn.Linear(128,128)
        self.lin2 = nn.Linear(128,1)

    def forward(self, g):
        h = self.gat1(g, g.ndata['final_hidden'])
        h = F.relu(h)
        h = h.reshape(-1,640)
        h = self.gat2(g, h)
        h = F.relu(h)
        g.ndata['h'] = h
        h = dgl.readout_nodes(g,'h',op='max')
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = F.relu(h)

        return h.flatten()