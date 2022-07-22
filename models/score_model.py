import dgl
from dgl.nn import GATConv
from dgl.nn import MaxPooling
import torch
import torch.nn as nn
import torch.nn.functional as F

lindropout = 0.2
attdropout = 0
featdropout = 0.05

class GAT1(nn.Module):
    def __init__(self):
        super(GAT1, self).__init__()
        self.gat1 = GATConv(64, 64, 10, allow_zero_in_degree=True, feat_drop=featdropout, attn_drop=attdropout)
        self.gat2 = GATConv(64,128,1, allow_zero_in_degree=True, feat_drop=featdropout, attn_drop=attdropout)
        self.maxpool = MaxPooling()
        self.lin1 = nn.Linear(128,128)
        self.lin2 = nn.Linear(128,1)
        self.dropout = nn.Dropout(lindropout)

    def forward(self, g):
        h = self.gat1(g, g.ndata['final_hidden'])
        h = F.relu(h)
        h = h.sum(axis=1)
        h = self.gat2(g, h)
        h = F.relu(h)
        g.ndata['h'] = h
        h = dgl.readout_nodes(g,'h',op='max')
        h = self.lin1(h)
        h=self.dropout(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = F.relu(h)

        return h.flatten()

class GAT2(nn.Module):
    def __init__(self):
        super(GAT2, self).__init__()
        self.gat1lig = GATConv(64, 64, 10, allow_zero_in_degree=True, feat_drop=featdropout, attn_drop=attdropout)
        self.gat2lig = GATConv(64,128,1, allow_zero_in_degree=True, feat_drop=featdropout, attn_drop=attdropout)
        self.gat1rec = GATConv(64, 64, 10, allow_zero_in_degree=True, feat_drop=featdropout, attn_drop=attdropout)
        self.gat2rec = GATConv(64,128,1, allow_zero_in_degree=True, feat_drop=featdropout, attn_drop=attdropout)
        self.maxpool = MaxPooling()
        self.lin1 = nn.Linear(256,128)
        self.lin2 = nn.Linear(128,1)
        self.dropout = nn.Dropout(lindropout)

    def forward(self, lig, rec):

        hlig = self.gat1lig(lig, lig.ndata['final_hidden'])
        hlig = F.relu(hlig)
        hlig = hlig.sum(axis=1)
        hlig = self.gat2lig(lig, hlig)
        hlig = F.relu(hlig)
        lig.ndata['h'] = hlig
        hlig = dgl.readout_nodes(lig,'h',op='max')

        hrec = self.gat1rec(rec, rec.ndata['final_hidden'])
        hrec = F.relu(hrec)
        hrec = hrec.sum(axis=1)
        hrec = self.gat2rec(rec, hrec)
        hrec = F.relu(hrec)
        rec.ndata['h'] = hrec
        hrec = dgl.readout_nodes(rec,'h',op='max')

        h = torch.cat((hlig,hrec),axis=2)
        h = h.squeeze(axis=1)

        h = self.lin1(h)
        h = self.dropout(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = F.relu(h)

        return h.flatten()