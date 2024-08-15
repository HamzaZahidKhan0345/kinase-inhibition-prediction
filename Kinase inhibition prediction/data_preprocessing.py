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
seed = 120
torch.manual_seed(seed)  
if torch.cuda.is_available():  
    device = "cuda:1"
    torch.cuda.manual_seed_all(seed)
    print("cuda:1")
else:  
    device = "cpu" 
    print(torch.cuda.is_available())
def createData(path,filename,datasetname,Smiles,Class):
    
    # DATA_PATH="/home/hamza/project/data/p450inh_final_public_80_20_1A2_train.txt"
    # df=pd.read_csv(DATA_PATH,sep='\t')
    df_test = pd.read_csv(path + '/' + filename,sep='\t')
    print(df_test.head(5))
    print(df_test.columns)
    # label_counts = df_test['ACT'].value_counts()
    # print(label_counts)
    smiles_data = df_test[Smiles]
    
    labels_data = df_test[Class]
    labels_data = labels_data.to_numpy()
    # smile_graph_test = {}
    solubility_arr_test = []
    smiles_array_test = []

    for i,smile in enumerate(smiles_data):
        solubility_arr_test.append(smiles_data[i])
        smiles_array_test.append(smile)
    
   
    noveltest_data = Molecule_data(root='Data/', dataset=datasetname,
                                    y=labels_data,smiles=smiles_array_test,
                                    transform=None)
    return noveltest_data

class Molecule_data(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='kinase', y=None, transform=None,
                  pre_transform=None,smiles=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(Molecule_data, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(y,smiles)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        # print('_process function called.')
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
    def process(self, y,smiles):
        
        
        data_list = []
        data_len = len(y)
        
        # featurizer = dc.feat.Mol2VecFingerprint()
        
        
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smile = smiles[i]
            label = y[i]                    
            
            
            data = from_smiles(smile)
            #print(data)
            data.x = (data.x).type(torch.FloatTensor)
            # print(data.x)
            data.edge_attr = (data.edge_attr).type(torch.FloatTensor)
            #print(data.edge_attr)
            data.y = torch.LongTensor([label])
            #print(data.y)
            graph = data
            data_list.append(graph)
            first_graph=(data_list[0])
        
            
        if self.pre_filter is not None:
              data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
          # save preprocessed data:
        print("self.processed_paths[0] : ", self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])


















