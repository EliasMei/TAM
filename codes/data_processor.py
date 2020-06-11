import torch
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
import pickle
import torch.utils.data

def standardization(templist_data):
    """ standarized templist_data
    Args:
        templist_data: a list of edge temp_list from several graphs
    """
    temp_data = np.array([i for item in templist_data for i in item])
    # normalize
    t_min = temp_data.min() - 1e-5
    t_max = temp_data.max() + 1e-5
    if t_max == t_min:
        raise ValueError('nan error')
    std_temp_data = (temp_data - t_min) / (t_max - t_min)
    std_templist_data = []
    ix = 0
    for item in templist_data:
        std_templist_data.append(std_temp_data[ix:ix+len(item)])
        ix = ix+len(item)
    return std_templist_data
    

def load_motionsense_data(path, attributes):
    times = ['time']
    data = pd.read_csv(path)[times+attributes]
    data = data.dropna()
    for attr in attributes:
        data[attr] = data[attr].apply(lambda x: round(x, 5))
    return data

def get_templist_dataset(graphs, max_len=200):
    """ 
    Args:
        graphs: a list of graphs 
        max_len: max length of temp_list of edge
    
    Returns: TensorDataset object for all edges in graph
    """
    templist_data = []
    for graph in graphs:
        node_values = list(graph.nodes.keys())
        for vi_ix, value_i in enumerate(node_values):
            for value_j in node_values[vi_ix+1:]:
                if (value_i, value_j) in graph.edges:
                    templist_data.append(graph.edges[(value_i, value_j)].temp_list)
    std_templist_data = standardization(templist_data)
    print(f'max_len of temp_list: {max_len}')
    with open('../output/excavator/temp_list.m', 'wb') as f:
        pickle.dump(std_templist_data,f)
    # raise ValueError
    templist_padded = np.array([np.pad(item, ((0, max_len-len(item))), 'constant') if max_len > len(item) else item[:max_len] for item in std_templist_data])
    templist_tensor = torch.from_numpy(templist_padded).unsqueeze(2).float()
    return templist_tensor

def get_node_tensor(graphs):
    """get node embedding ix data
    
    For each node, we assign a ix number. If two nodes in different graphs have the same value, they are still asiigned different ix number. 

    Args:
        graphs: a list of graphs
    
    Returns:
        nodes_dataset: a TensorDataset object for all pairs of nodes in each edge
    """
    nodes = [node for graph in graphs for _, node in graph.nodes.items()]
    node_ix_dict = {node:ix for ix, node in enumerate(nodes)}
    edges = []
    for graph in graphs:
        graph_node_values = list(graph.nodes.keys())
        graph_edges = []
        for vi_ix, value_i in enumerate(graph_node_values):
            for value_j in graph_node_values[vi_ix+1:]:
                if (value_i, value_j) in graph.edges:
                    graph_edges.append((value_i, value_j))
        edges.append(graph_edges)
    nodes_data = [[node_ix_dict[graph.nodes[vi]], node_ix_dict[graph.nodes[vj]]] for graph_ix, graph in enumerate(graphs) for vi, vj in edges[graph_ix]]
    node_tensor = torch.from_numpy(np.array(nodes_data)).long()
    return node_tensor

def get_graph_dataloader(graphs, max_len=200, batch_size=128, shuffle=False):
    node_tensor = get_node_tensor(graphs)
    templist_tensor = get_templist_dataset(graphs, max_len=max_len)
    graph_dataset = torch.utils.data.TensorDataset(node_tensor, templist_tensor)
    graph_dataloader = torch.utils.data.DataLoader(graph_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return graph_dataloader

def get_templstm_tensor(norm_data, seq_len, predict_len):
    seq_data = []
    label_data = []
    for ix in range(0, len(norm_data) - seq_len - predict_len+1, seq_len):
        input_ = norm_data[ix:ix + seq_len]
        seq_data.append(input_)
        label = norm_data[ix + seq_len:ix + seq_len + predict_len]
        label_data.append(label)
    seq_tensor = torch.from_numpy(np.array(seq_data)).unsqueeze(2)
    label_tensor = torch.from_numpy(np.array(label_data))
    return seq_tensor, label_tensor

def get_templstm_dataloaders(graphs, times=None, attributes=None, seq_len=20, predict_len=20, batch_size=128):
    """generate a dataloader for temp_lstm

    Temp_LSTM model takes the previous seq_len values to predict the (seq_len + 1) value
    Before generating dataloders, we need to standardize attribute data respectively.

    Args:
        graphs: a list of graph
        seq_len: use the number of "seq_len" values to predict the next "pre_dict" values
        predict_len: use the number of "seq_len" values to predict the next "pre_dict" values
        batch_size: batch size of the dataloader
    
    Returns:
        dataloders:= a list of dataloader where each dataloader represent an attribute of a csv (attributes contain time and value attributes)
            e.g.: [[csv1.attr1.dataloader, csv1.attr2.dataloader, ...], [csv2.attr1.dataloader, csv2.attr2.dataloader, ...]]
    """
    dataloaders = []
    def normalize(attr_data):
        data = np.array([i for item in attr_data for i in item])
        data_min = data.min()
        data_max = data.max()
        std_attr_data = (data - data_min) / (data_max - data_min)
        std_attr_data = std_attr_data.astype(np.float32)
        return std_attr_data

    total_attributes = times + attributes
    # norm_templstm_data: [[attr1.csv1.norm_data, attr1.csv2.norm_data,...], [attr2.csv1.norm_data, attr2.csv2.norm_data,...], ...]
    norm_templstm_data = []
    for attr in total_attributes:
        attr_data = []
        for csv in graphs:
            data = csv[0][0].data
            attr_data.append(data[attr])
        norm_attr_data = normalize(attr_data)
        norm_templstm_data.append(norm_attr_data)

    ix = 0
    for csv_ix, csv in enumerate(graphs):
        csv_dataloaders = []
        data_len = len(csv[0][0].data)
        for attr_ix, attr in enumerate(total_attributes):
            norm_data = norm_templstm_data[attr_ix][ix:ix+data_len]
            seq_tensor, label_tensor = get_templstm_tensor(norm_data, seq_len, predict_len)
            templstm_dataset = torch.utils.data.TensorDataset(seq_tensor, label_tensor)
            data_loader = torch.utils.data.DataLoader(templstm_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            csv_dataloaders.append(data_loader)
        ix += data_len
        dataloaders.append(csv_dataloaders)
    return dataloaders
