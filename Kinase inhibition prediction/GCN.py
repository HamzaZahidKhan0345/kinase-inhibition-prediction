#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:44:30 2024

@author: hamza
"""

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch.nn.functional import relu 
import torch.nn as nn
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN,self).__init__()
        torch.manual_seed(12345)
        self.gcn_conv1 = GCNConv(9,hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.gcn_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.lin = Linear(hidden_channels, 1)
        
    def forward(self, x, edge_index, batch):
        x = self.gcn_conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.gcn_conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x= global_max_pool(x,batch)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x
    
    
    
    