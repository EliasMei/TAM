import torch
import random
import pickle
import pandas
import time
import numpy as np
from pandas import DataFrame
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_processor import *


class Seq2Vec(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Seq2Vec, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = 1

        # the input of lstm is a three-dim tensor (seq, batch, feature-dim)
        self.lstm = nn.LSTM(input_size=self.input_size, batch_first=True,
                            hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        lstm_out, (hidden, context) = self.lstm(input_data)
        output = self.sigmoid(self.fc(lstm_out))
        return output, hidden


class TempLSTM(nn.Module):
    """Baseline: LSTM

    Temp_LSTM model takes the previous seq_len values to predict the (seq_len + 1) value

    Args:
        graphs: a list of graphs
        seq_len: use the number of "seq_len" values to predict the next "pre_dict" values
        predict_len: use the number of "seq_len" values to predict the next "pre_dict" values
        hidden_size: the dimension of hidden
        batch_size: batch size of the dataloader
        max_epoch: max train epoch
    """

    def __init__(self, graphs, times=None, attributes=None, seq_len=100, predict_len=100, hidden_size=100, batch_size=32, max_epoch=150, pre_trained=False, path='../output/model/temp_lstm.pt'):
        super(TempLSTM, self).__init__()
        self.graphs = graphs
        self.times = times
        self.attributes = attributes
        self.total_attributes = times+attributes
        self.path = path
        self.seq_len = seq_len
        self.predict_len = predict_len
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.dataloaders = get_templstm_dataloaders(
            graphs, times=times, attributes=attributes, seq_len=self.seq_len, predict_len=self.predict_len, batch_size=self.batch_size)
        self.model = Seq2Vec(1, hidden_size, batch_size).cuda()
        if pre_trained:
            self.model.load_state_dict(torch.load(path)['state_dict'])
        else:
            self.model = self.train(self.model)

        self.vector_dic = self.extract_vec()
        self.similarity_dict = self.generate_similarity_dict()

    def train_epoch(self, criterion, optimizer):
        for csv_data_loaders in self.dataloaders:
            for data_loader in csv_data_loaders:
                cost = 0
                for batch_data in data_loader:
                    self.model.zero_grad()
                    seq_data, label_data = batch_data
                    seq_data = seq_data.cuda()
                    label_data = label_data.cuda()
                    output, _ = self.model(seq_data)
                    output = output.squeeze()
                    loss = criterion(output, label_data)
                    cost += loss.item()
                    loss.backward()
                    optimizer.step()
        return cost

    def train(self, model):
        model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        loss_list = []
        for epoch in range(self.max_epoch):
            start_time = time.time()
            epoch_loss = self.train_epoch(criterion, optimizer)
            scheduler.step(epoch_loss)
            loss_list.append(epoch_loss)
            end_time = time.time()
            if epoch == 0 or (epoch+1) % 10 == 0:
                print(
                    f'Model: TempLSTM - Epoch: {epoch + 1}/{self.max_epoch} - Loss: {epoch_loss} - Time: {round(end_time-start_time, 2)}s')
        torch.save({
            'epoch': self.max_epoch,
            'state_dict': model.state_dict(),
        }, self.path)
        print(f'model saved at {self.path}')
        return model

    def extract_vec(self):
        self.model.eval()
        vector_dic = dict()
        for csv_ix, csv_data_loaders in enumerate(self.dataloaders):
            data_file_name = self.graphs[csv_ix][0][0].file_name
            vector_dic[data_file_name] = {}
            for attr_ix, data_loader in enumerate(csv_data_loaders):
                attr = self.total_attributes[attr_ix]
                seq_vectors = []
                for batch_data in data_loader:
                    self.model.zero_grad()
                    seq_data, _ = batch_data
                    seq_data = seq_data.cuda()
                    _, hidden = self.model.forward(seq_data)
                    seq_vec = hidden.view(-1, self.hidden_size)
                    seq_vectors.append(seq_vec.cpu().detach().numpy())
                vectors = np.concatenate(seq_vectors, axis=0)
                vector = np.mean(vectors, axis=0).squeeze()
                vector_dic[data_file_name][attr] = vector
        return vector_dic

    def generate_similarity_dict(self):
        similarity_dict = {}
        total_attributes = self.times + self.attributes
        for csv_ix_1, csv_graphs_1 in enumerate(self.graphs):
            data_file_name_1 = csv_graphs_1[0][0].file_name
            for csv_ix_2, csv_graphs_2 in enumerate(self.graphs[csv_ix_1+1:]):
                data_file_name_2 = csv_graphs_2[0][0].file_name
                for attr_1 in total_attributes:
                    key_1 = data_file_name_1 + attr_1
                    vec_1 = self.vector_dic[data_file_name_1][attr_1]
                    for attr_2 in total_attributes:
                        vec_2 = self.vector_dic[data_file_name_2][attr_2]
                        key_2 = data_file_name_2 + attr_2
                        similarity = np.dot(
                            vec_1, vec_2) / (np.linalg.norm(vec_1)*np.linalg.norm(vec_2))
                        similarity_dict[(key_1, key_2)] = similarity
                        similarity_dict[(key_2, key_1)] = similarity
        return similarity_dict

    def similarity(self, data_file_name_1, data_file_name_2, attr_1, attr_2):
        key_1 = data_file_name_1 + attr_1
        key_2 = data_file_name_2 + attr_2
        return self.similarity_dict[(key_1, key_2)]
