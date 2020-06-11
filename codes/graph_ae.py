import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LSTM_HIDDEN_SIZE = 128
DENSE_HIDDEN_SIZE = 2 * LSTM_HIDDEN_SIZE
GRU_HIDDEN_SIZE = 50

class Graph_AutoEncoder(nn.Module):
    """An auto-encoder model used for embedding nodes in all graphs at the same time
    
    This model train all nodes and edges of graphs at the same time, i.e embedding nodes from different graphs at the same time. Actually, this model did not take advantage any structual information of grpahs. It just take a triple (head_node, tail_node, temp_list) as the input, and optimize the model by minimizing the temp_list re-construct error. The node embedding concats the temp_list embedding and the concatenate vector is used for re-constructing temp_list.

    Args:
        node_num: total number of nodes
        node_emb_size: the dimension of node embedding
        hidden_size: the dimension of temp_list embedding
    
    """
    def __init__(self, hidden_size, init_node_embedding):
        super(Graph_AutoEncoder, self).__init__()
        self.node_num, self.node_emb_size = init_node_embedding.shape
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.bidirectional = True

        self.node_embeddings = nn.Embedding(self.node_num, self.node_emb_size)
        self.node_embeddings.weight.data.copy_(init_node_embedding)

        # encoder network layers
        self.encoder_lstm_1 = nn.LSTM(1,hidden_size=LSTM_HIDDEN_SIZE, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True,dropout=0.5)
        self.encoder_lstm_2 = nn.LSTM(2*LSTM_HIDDEN_SIZE,hidden_size=LSTM_HIDDEN_SIZE, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=0.5)

        self.encoder_fc_1 = nn.Linear(DENSE_HIDDEN_SIZE, out_features=DENSE_HIDDEN_SIZE)
        self.encoder_fc_2 = nn.Linear(DENSE_HIDDEN_SIZE, out_features=hidden_size)

        self.decoder_cell_1 = nn.GRUCell(1, self.hidden_size + self.node_emb_size)
        self.decoder_cell_2 = nn.GRUCell(self.hidden_size + self.node_emb_size, GRU_HIDDEN_SIZE)

        self.decoder_fc = nn.Linear(GRU_HIDDEN_SIZE, 1)

        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()
    
    def encoder(self, node_data, edge_data):
        """encode temp_list, head node and tail nodes into one vector
        
        The encoder first transform the temp_list into one vector, and concat this vector with the embeddings of the head node and tail node.

        Args:
            node_data: a batch of nodes' ix. Each row in the data denotes a pair (head_node ix, tail_node ix) which are the head node and tail node of the corresponding edge. shape: (batch_size, 2)
            edge_data: a batch of temp_list. shape: (batch_size, max_len) where max_len denotes the maximal length of temp_list
        """
        # node_data.shape: (batch_size, 2)
        node_embeddings = self.node_embeddings(node_data.long())
        node_1_emb = node_embeddings[:,0,:]
        node_2_emb = node_embeddings[:,1,:]
        node_emb = (node_1_emb + node_2_emb) / 2
        h_lstm_1, (h_n_1, c_n_1) = self.encoder_lstm_1(edge_data)
        _, (h_n_2, c_n_2) = self.encoder_lstm_2(h_lstm_1)
        h_n_1 = h_n_1.view(self.num_layers, 2, -1, LSTM_HIDDEN_SIZE)
        h_n_2 = h_n_2.view(self.num_layers, 2, -1, LSTM_HIDDEN_SIZE)
        avg_h_n_1 = torch.mean(h_n_1, 0)
        h_n_1_conc = avg_h_n_1.view(-1, 2*LSTM_HIDDEN_SIZE)
        avg_h_n_2 = torch.mean(h_n_2, 0)
        h_n_2_conc = avg_h_n_2.view(-1, 2*LSTM_HIDDEN_SIZE)
        hidden_lstm = h_n_1_conc + h_n_2_conc
        h_fc_1 = self.dropout(self.sigmoid(self.encoder_fc_1(hidden_lstm)))
        temp_list_emb = self.sigmoid(self.encoder_fc_2(h_fc_1))
        hidden = torch.cat((temp_list_emb, node_emb), 1)
        return hidden

    def decoder_step(self, prev_result, prev_hidden_1, prev_hidden_2=None):
        hidden_1 = self.decoder_cell_1(prev_result, prev_hidden_1)
        if prev_hidden_2 is not None:
            hidden_2 = self.decoder_cell_2(hidden_1, prev_hidden_2)
        else:
            hidden_2 = self.decoder_cell_2(hidden_1)
        result = self.sigmoid(self.decoder_fc(hidden_2))
        return hidden_1, hidden_2, result

    def decoder(self, hidden, edge_data):
        max_len = edge_data.shape[1]
        results = []
        hidden_1, hidden_2, result = self.decoder_step(edge_data[:,-1], hidden)
        results.append(result)
        for ix in range(max_len-1):
            hidden_1, hidden_2, result = self.decoder_step(result, hidden_1, hidden_2)
            results.append(result)
        output = torch.cat(results, 1).unsqueeze(2)
        return output

    def forward(self, node_data, edge_data):
        hidden = self.encoder(node_data, edge_data)
        output = self.decoder(hidden, edge_data)
        return output