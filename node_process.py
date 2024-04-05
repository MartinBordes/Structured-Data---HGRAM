import numpy as np
import pandas as pd
import pickle
from dgl.data import *
import argparse
import torch
from pathlib import Path

def main():

    # this is an example of disjoint label multiple graphs.
    dataset_name = args.dataset.__name__
    dataset = args.dataset()
    
    # assume you have a list of DGL graphs stored in the variable dgl_Gs
    if len(dataset) == 1:
        dgl_Gs = [dataset[0]]
    else:
        dgl_Gs = list(dataset)
    # assume you have an array of features where [feat_1, feat_2, ...] and each feat_i corresponding to the graph i.
    feature_map = [np.array(graph.ndata['feat']) for graph in dgl_Gs]
    # assume you have an array of labels where [label_1, label_2, ...] and each label_i corresponding to the graph i.
    label_map = [np.array(graph.ndata['label']) for graph in dgl_Gs]
    
    info = {}
    
    for idx, G in enumerate(dgl_Gs):    
        # G is a dgl graph
        for j in range(len(label_map[idx])):
            info[str(idx) + '_' + str(j)] = label_map[idx][j]
                
    df = pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={"index": "name", 0: "label"})

    train_df_list = []
    val_df_list = []
    test_df_list = []
    for idx, G in enumerate(dgl_Gs):
        if G.ndata['train_mask'].shape[1] > 1:
            train_ind = [index.item() for index in torch.nonzero(G.ndata['train_mask'][:,0])[:,0]]
            train = df[df.name.str.contains(str(idx)+'_'+str(train_ind))]
            train_df_list.append(train)
            
            val_ind = [index.item() for index in torch.nonzero(G.ndata['val_mask'][:,0])[:,0]]
            val = df[df.name.str.contains(str(idx)+'_'+str(val_ind))]
            val_df_list.append(val)
            
            test_ind = [index.item() for index in torch.nonzero(G.ndata['test_mask'][:,0])[:,0]]
            test = df[df.name.str.contains(str(idx)+'_'+str(test_ind))]
            test_df_list.append(test)
        else:
            train_ind = [index.item() for index in torch.nonzero(G.ndata['train_mask'])]
            train = df[df.name.str.contains(str(idx)+'_'+str(train_ind))]
            train_df_list.append(train)
            
            val_ind = [index.item() for index in torch.nonzero(G.ndata['val_mask'])]
            val = df[df.name.str.contains(str(idx)+'_'+str(val_ind))]
            val_df_list.append(val)
            
            test_ind = [index.item() for index in torch.nonzero(G.ndata['test_mask'])]
            test = df[df.name.str.contains(str(idx)+'_'+str(test_ind))]
            test_df_list.append(test)
        
    train_df = pd.concat(train_df_list)
    val_df = pd.concat(val_df_list)
    test_df = pd.concat(test_df_list)

    directory = Path('data/' + str(dataset_name))
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    train_df.reset_index(drop = True).to_csv('data/' + str(dataset_name) + '/train.csv')
    val_df.reset_index(drop = True).to_csv('data/' + str(dataset_name) + '/val.csv')
    test_df.reset_index(drop = True).to_csv('data/' + str(dataset_name) + '/test.csv')
    
    with open('data/' + str(dataset_name) + '/graph_dgl.pkl', 'wb') as f:
        pickle.dump(dgl_Gs, f)
        
    with open('data/' + str(dataset_name) + '/label.pkl', 'wb') as f:
        pickle.dump(info, f)
        
    np.save('data/' + str(dataset_name) + '/features.npy', np.array(feature_map))
    

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=type, help='dataset Name (possibilities : CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset... (see https://docs.dgl.ai/en/2.0.x/api/python/dgl.data.html#node-prediction-datasets)', default=CoraGraphDataset)

    args = argparser.parse_args()

    main()