"""Adapted From: https://github.com/tkipf/pygcn/blob/master/pygcn/models.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import GraphConvolution

import numpy as np

class GCN(nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    
    def loss(self, probs, y, loss_idx):
        return F.nll_loss(probs[loss_idx], y[loss_idx])
    
class GCNMod(nn.Module):
    """Graph Convolutional Network with modified loss function to incorporate network community structure"""

    def __init__(self, nfeat, nhid, nclass, dropout, adj, weight=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
        self.weight = weight
        
        A = adj.toarray()
        D = sum(A)
        e = np.sum(A) / 2
        B = A  - np.outer(D,D)/(2*e)

        self.e = e
        self.B = torch.FloatTensor(B)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    
    def loss(self, probs, y, loss_idx):
        loss_train = F.nll_loss(probs[loss_idx], y[loss_idx])
        softmax = F.softmax(probs, dim=1)
        loss_mod = torch.trace(torch.matmul(torch.matmul(softmax.T, self.B), softmax)) * (1/(2*self.e))
        loss = ((1-self.weight) * loss_train) -  (self.weight * loss_mod)
        return loss
    
class GCNAux(nn.Module):
    """Alternative form of GCNMod where a different 2nd layer is used to calculate the modularitiy optimization term, the first layer is shared"""

    def __init__(self, nfeat, nhid, nclass, dropout, adj, weight=0.5):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.aux = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.weight = weight
        
        A = adj.toarray()
        D = sum(A)
        e = np.sum(A) / 2
        B = A  - np.outer(D,D)/(2*e)

        self.e = e
        self.B = torch.FloatTensor(B)

    def forward_int(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

    def forward_aux(self, x, adj):
        x = self.forward_int(x, adj)
        x = self.aux(x, adj)
        return x

    def forward(self, x, adj):
        x = self.forward_int(x, adj)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    
    def loss(self, probs, h_aux, y, loss_idx):
        loss_train = F.nll_loss(probs[loss_idx], y[loss_idx])
        h_aux = F.softmax(h_aux, dim=1)
        loss_mod = torch.trace(torch.matmul(torch.matmul(h_aux.T, self.B), h_aux)) * (1/(2*self.e))
        loss = ((1-self.weight) * loss_train) -  (self.weight * loss_mod)
        return loss
