import dgl
from dgl.nn import GATConv
from dgl.nn import MaxPooling
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.gat1 = GATConv(64, 64, 10)
        self.gat2 = GATConv(64,128,1)
        self.maxpool = MaxPooling()
        self.lin1 = nn.Linear(128,128)
        self.lin2 = nn.Linear(128,1)

    def forward(self, g, in_feat):
        h = self.gat1(g, in_feat)
        h = F.relu(h)
        h = h.mean(axis=1)
        h = self.gat2(g, h)
        h = F.relu(h)
        h = self.maxpool(g, h)
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = F.relu(h)

        return h