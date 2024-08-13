import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import numpy as np
import torch.nn.functional as F
import os
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset

from torch_geometric.loader import DataLoader

from torch_geometric.utils.smiles import from_smiles

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch.nn import Linear

import torch.nn.functional as F
import os
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool 
from torch_geometric.nn import graclus
from torch_geometric.nn import global_add_pool
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix,balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score   
criterion = torch.nn.BCELoss()
def train(model,optimizer,train_loader,device):
    model.train()
    correct  = 0
    total = 0
    loss_sum = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    total_loss= 0
    for data in train_loader:# Iterate in batches over the training dataset.
        # print(data.y.shape)
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)# Perform a single forward pass.
        
       
        labels = data.y.float()
       
        labels = torch.unsqueeze(labels, 1)
        # converting the labels to numpy 
        
        
        loss = criterion(out, labels)  # Compute the loss.
        optimizer.zero_grad() # Clear gradients. 
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
        #total_loss += float(loss) * data.num_graphs
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        
        
        
 
        predicted = torch.round(out)     # class with the highest probability is selected as the predicted class label
        # predicted = predicted.cpu().detach().numpy()
        correct += (predicted == labels).sum().item()  
        total += labels.size(0)   # total will contain the sum of the number of samples in each batch, 

        loss_sum += loss.item()
        
        # precision = precision_score(labels,predicted)
        # recall = recall_score(labels, predicted)
        # f1 = f1_score(labels, predicted)
        # mcc = matthews_corrcoef(labels, predicted)
        # balanced_accuracy = balanced_accuracy_score(labels, predicted)
        # fpr, tpr, thresholds = roc_curve(labels,predicted)
        # auc_roc = roc_auc_score(labels, predicted)
        # sensitivity, specificity = calculate_sensitivity_specificity(labels, predicted)
        
        tp += ((predicted == 1) & (labels == 1)).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()
         
        accuracy = correct / total      # total will contain the sum of the number of samples in each batch,
        avg_loss = loss_sum / len(train_loader)
        mcc = calculate_mcc(tp, tn, fp, fn)
        ba = calculate_ba(tp, tn, fp, fn)
        
        sensitivity, specificity, precision, f1_score = calculate_sensitivity_specificity(tp, tn, fp, fn)
        
    return accuracy, avg_loss, mcc, ba, sensitivity, specificity  
    

@torch.no_grad()  # works same like model.eval
def test(model,test_loader,device):
    correct = 0
    total = 0
    loss_sum = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    mse = []
    total_loss = 0
    # model.eval()
    # dataloader=len(test_loader)
    for data in test_loader:
        #print(data[0].y)
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        # print(data.batch)
        labels = data.y.float()
        labels = torch.unsqueeze(labels, 1)
        loss = criterion(out, labels)  # Compute the loss.
        loss_sum += loss.item()
    
        predicted = torch.round(out)
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # precision = precision_score(labels,predicted)
        # recall = recall_score(labels, predicted)
        # f1 = f1_score(labels, predicted)
        # mcc = matthews_corrcoef(labels, predicted)
        # balanced_accuracy = balanced_accuracy_score(labels, predicted)
        # fpr, tpr, thresholds = roc_curve(labels,predicted)
        # auc_roc = roc_auc_score(labels, predicted)
        
        
        tp += ((predicted == 1) & (labels == 1)).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()
         
        accuracy_test = correct / total      # total will contain the sum of the number of samples in each batch,
        avg_loss_test = loss_sum / len(test_loader)
        mcc_test = calculate_mcc(tp, tn, fp, fn)
        ba_test = calculate_ba(tp, tn, fp, fn)
        sensitivity_test, specificity_test, precision_test, f1_score_test  = calculate_sensitivity_specificity(tp, tn, fp, fn)
    return accuracy_test, avg_loss_test, mcc_test, ba_test, sensitivity_test, specificity_test, precision_test, f1_score_test
        
        
@torch.no_grad()
def predicting(loader, model, device):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        y = data.y.view([-1])
        out1 = out.view([-1])
        # print("test : ", y.shape)
        test_loss = F.mse_loss(out1, y)
        # print("no of graphs: ", data.num_graphs)
        total_loss += float(test_loss) * data.num_graphs
        total_examples += data.num_graphs
        total_preds = torch.cat((total_preds, out.view(-1, 1).cpu()), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    return total_loss,sqrt(total_loss / total_examples),total_labels.numpy().flatten(),total_preds.numpy().flatten()
    
        
def calculate_mcc(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = numerator / (denominator + 1e-8)  # Add small epsilon to avoid division by zero
    return mcc


def calculate_ba(tp, tn, fp, fn):
    sensitivity = tp / ((tp + fn)+ 1e-8)
    specificity = tn / ((tn + fp)+ 1e-8)
    ba = (sensitivity + specificity) / 2
    return ba
 

def calculate_sensitivity_specificity(tp, tn, fp, fn):
    
    sensitivity = tp / ((tp + fn)+ 1e-8)
    specificity = tn / ((tn + fp)+ 1e-8)
    precision = tp / ((tp + fp) + 1e-8)
    recall = tp / ((tp + fn) + 1e-8)
    f1_score = 2 * (precision * recall) / ((precision + recall + 1e-8))
    return sensitivity, specificity, precision, f1_score




