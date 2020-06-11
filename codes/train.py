import os
import math
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from tqdm._tqdm_notebook import tnrange
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from temp_graph import *
from graph_ae import *


def train_epoch(data_loader, model, criterion, optimizer):
    cost = 0
    for ix, batch_data in enumerate(data_loader):
        model.zero_grad()
        node_data, temp_data = batch_data
        node_data = node_data.cuda()
        temp_data = temp_data.cuda()
        output = model(node_data, temp_data)
        loss = criterion(output, temp_data)
        cost += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
    return cost

def train(model, data_loader, path, init_node_embedding, hidden_size=100, max_epoch=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_num, node_emb_size = init_node_embedding.shape
    model = model.cuda()
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    loss_list = []
    for epoch in range(max_epoch):
        start_time = time.time()
        epoch_loss = train_epoch(data_loader, model, criterion, optimizer)
        if math.isnan(epoch_loss):
            print(f'retrain. Loss: {epoch_loss}')
            return None, None
            # raise ValueError('Nan Error')
        scheduler.step(epoch_loss)
        loss_list.append(epoch_loss)
        end_time = time.time()
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(
                f'Epoch: {epoch + 1}/{max_epoch} - Loss: {epoch_loss} - Time: {round(end_time-start_time, 2)}s') 

    torch.save({
        'epoch': max_epoch,
        'state_dict': model.state_dict(),
    }, path)
    print(f'model saved at {path}')
    return model, loss_list
