import os
import numpy as np
import pandas as pd
from sklearn import metrics 
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from tqdm.notebook import tqdm
from data_preparation.data_preprocessing import Molecule_data,createData
from model.gcn import GCN_GAT
from model.classification_train_test import train, test, predicting
from folds import folds_creator_dataset_2
import numpy as np
import warnings



warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
seed = 43
torch.manual_seed(seed)  
if torch.cuda.is_available():  
    device = "cuda:1"
    torch.cuda.manual_seed_all(seed)
    print("cuda:1")
else:  
    device = "cpu" 
    print(torch.cuda.is_available())
NO_OF_FOLDS = 10
folds = 10
results = []
best_rmse_arr = []
bestrmsesum = 0
scores = []
true_val = []
pred_val = []
NUM_EPOCHS = 1000

savepath = 'GCN_GAT/'
def readAndCreateData(df):
    
    # df = pd.read_csv('Data_Prep/solubility_1.csv')
    df = df.sample(frac=1).reset_index(drop=True)  #  useful for randomizing the order of data before splitting it into training and validation sets
    activity = df['Class']           # label columns
    activity = activity.to_numpy()
    num_positive = np.sum(activity == 1)
    num_negative = np.sum(activity == 0)

    print("Number of positive samples:", num_positive)
    print("Number of negative samples:", num_negative)
    
    processed_data_file_train = 'Data_2/'+ savepath +'processed/train_data_set_fold_'+str(0)+'.pt'
    processed_data_file_test = 'Data_2/'+ savepath + 'processed/test_data_set_fold_'+str(0)+'.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data.py to prepare data in pytorch format!')
            folds_creator_dataset_2.createFoldsCsv(df, NO_OF_FOLDS, 'Smiles', 'Class', savepath=savepath)                 
def trainfolds():
    device = "cuda:1"
    
    for fold in tqdm(range(NO_OF_FOLDS)):
        
        
      
        model = GCN_GAT(hidden_channels=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        criterion = torch.nn.BCELoss()

        device = "cuda:1"

    

        
        test_losses = []
        train_losses = []
        patience = 20
        trigger_times = 0
        the_last_loss = 100
        if not os.path.exists('saved_models_dataset_2/'+savepath):
            os.makedirs('saved_models_dataset_2/'+savepath)
        model_file_name = 'saved_models_dataset_2/' + savepath +'model_' +  str(fold) +  '.model'
        train_data = Molecule_data(root='Data_2/'+savepath, dataset='train_data_set_fold_'+str(fold),
                                    y=None,smiles=None, transform=None)
        test_data = Molecule_data(root='Data_2/'+savepath, dataset='test_data_set_fold_'+str(fold),
                                  y=None,smiles=None, transform=None)
        
        train_loader=DataLoader(train_data,batch_size=32,shuffle=True)
        test_loader=DataLoader(test_data,batch_size=32,shuffle=True)
        best_ret = []
        
       
        for epoch in range(NUM_EPOCHS):
            accuracy, avg_loss, mcc, ba, sensitivity, specificity = train(model, optimizer, train_loader, device)
            accuracy_test, avg_loss_test, mcc_test, ba_test, sensitivity_test, specificity_test, precision_test, f1_score_test = test(model,test_loader,device)
            print(f'Epoch: {epoch:03d}')
            
            print(f'train accuracy: {accuracy:.4f}, train loss: {avg_loss:.4f}, '
              f'train MCC: {mcc:.4f}, train BA: {ba:.4f}, train Sn:{sensitivity:.4f}, train Sp:{specificity:.4f}')

            print(f'test accuracy: {accuracy_test:.4f}, test loss: {avg_loss_test:.4f}, '
              f'test MCC: {mcc_test:.4f}, test BA: {ba_test:.4f}, test Sn:{sensitivity_test:.4f}, test Sp:{specificity_test:.4f}, '
              f'test precision:{precision_test:.4f}, test recall:{f1_score_test:.4f}')
            
            
          
            
            
            train_losses.append(avg_loss)
            test_losses.append(avg_loss_test)
            the_current_loss = avg_loss_test   #.item()
            # best_ret.append(ret)
            if the_current_loss > the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)
        
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break
            else:
                # ret = [epoch,train_loss,test_loss] #, score
                trigger_times = 0
                the_last_loss = the_current_loss
                best_rmse = the_current_loss
                
                torch.save(model.state_dict(), model_file_name)
                
def main():
    df = pd.read_csv('/home/hamza/second_paper_pdgfrp/Data_2/train data_2.txt', sep='\t')
    
    
    
    readAndCreateData(df)
    results = trainfolds()
      

if __name__ == "__main__":
    main()
 




































