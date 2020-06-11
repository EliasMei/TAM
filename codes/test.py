import argparse
import math
import os
import pickle
import random
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_processor import *
from graph_matching import *
from temp_graph import *
from temp_lstm import *
from train import *
from util import *
from tam import *
from fod import *
from relation_graph import *

plt.switch_backend('agg')
tqdm.pandas()

parser = argparse.ArgumentParser(description='PyTorch Temporal Dependency Graph Embedding Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--max_len', type=int, default=200,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--dataset', default='wind', help='Which dataset do you want')
parser.add_argument('--intervals', type=int, default=20, help='The number of intervals')
parser.add_argument('--pre_trained', type=bool, default=False, help='pre_trained')
parser.add_argument('--pre_load', type=bool, default=False, help='pre_load')
parser.add_argument('--time_diff_threshold', type=int, default=600, help='time_diff_threshold')
parser.add_argument('--iter_num', type=int, default=1, help='the number of iteration')

args = parser.parse_args()
print(args)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def exp_turbo(graphs, times, attributes, dataset_name, method, node_model=None, edge_model=None):
    """
    Apply another graph matching for deal with multiple time attributes
    """
    # length of graph_matching_result is the number of total_attributes
    total_attributes = times + attributes
    graph_matching_result = np.zeros(len(total_attributes))
    count = 0

    model = RelationBipartiteGraph(graphs, times, attributes, dataset_name=dataset_name, method=method, node_model=node_model, edge_model=edge_model, random_module=random)
    matching_res = dict()

    for csv_ix_1, csv_1 in enumerate(graphs):
        for ix, csv_2 in enumerate(graphs[csv_ix_1+1:]):
            csv_ix_2 = csv_ix_1 + ix + 1
            count += 1
            match = model.get_match(csv_ix_1, csv_ix_2)
            for key, value in match.items():
                if key != value:
                    continue
                graph_matching_result[key] += 1
    return graph_matching_result / count, model.hist


def exp_epoch(graphs, times, attributes, dataset_name, time_graph_gen, epoch):
    """Experiments on accuracy over different attributes
    Test accuracy for each candidate attribute and average accuracy for the dataset.
    Args:
        graphs: a 3-layer list of graphs: [csv[time[attr.graph]]]
            e.x.:
            [[csv1[time_1[attr_1.graph, attr_2.graph, ...], time_2[...]], [csv1[time_1[attr_1.graph, attr_2.graph, ...], time_2[...], [...]]
    """
    methods = ['FOD', 'TempLSTM', 'TAM', 'FOD+TAM', 'TempLSTM+TAM']
    train_graphs = [graph for csv in graphs for time_graphs in csv for graph in time_graphs]

    fod_start = time.clock()
    fod = FOD(graphs, times, attributes)
    fod_time = time.clock() - fod_start

    lstm_start = time.clock()
    lstm = TempLSTM(graphs, times=times, attributes=attributes, max_epoch=100, seq_len=20, predict_len=20, pre_trained=False, path=f'../output/{dataset_name}/lstm_{epoch}.pt')
    lstm_time = time.clock() - lstm_start

    tam_start = time.clock()
    tam = TAM(train_graphs, max_epoch=150, max_len=args.max_len, batch_size=args.batch_size, pre_trained=args.pre_trained, path=f'../output/{dataset_name}/graph_ae_{epoch}.pt',node_emb_size=100, iter_num=args.iter_num)
    tam_time = time.clock() - tam_start + time_graph_gen

    graph_matching_results = []
    total_time_costs = np.zeros(len(methods))
    lstm_time_costs = np.zeros(len(methods))
    tam_train_time_costs = np.zeros(len(methods))
    tam_matching_time_costs = np.zeros(len(methods))
    for ix, method in enumerate(methods):
        total_time_cost = 0
        lstm_time_cost = 0
        tam_train_time_cost = 0
        tam_matching_time_cost = 0
        if method == 'FOD':
            total_time_cost += fod_time
            node_model = fod
            edge_model = None
        elif method == 'TempLSTM':
            total_time_cost += lstm_time
            lstm_time_cost += lstm_time
            node_model = lstm
            edge_model = None
        elif method == 'TAM':
            total_time_cost += tam_time
            tam_train_time_cost += tam.train_time
            tam_matching_time_cost += tam.matching_time
            node_model = None
            edge_model = tam
        elif method == 'FOD+TAM':
            total_time_cost += (fod_time + tam_time)
            tam_train_time_cost += tam.train_time
            tam_matching_time_cost += tam.matching_time
            node_model = fod
            edge_model = tam
        elif method == 'TempLSTM+TAM':
            total_time_cost += (lstm_time + tam_time)
            lstm_time_cost += lstm_time
            tam_train_time_cost += tam.train_time
            tam_matching_time_cost += tam.matching_time
            node_model = lstm
            edge_model = tam
        else:
            raise AttributeError('Invalid Baseline')
        turbo_start = time.clock()
        graph_matching_result, hist = exp_turbo(graphs, times, attributes,  dataset_name=dataset_name, method=method, node_model=node_model, edge_model=edge_model)
        time_turbo = time.clock() - turbo_start
        with open(f'../output/{dataset_name}/{method}_hist_{epoch}.json', 'w') as f:
            json.dump(hist, f)
        graph_matching_results.append(graph_matching_result)
        total_time_costs[ix] = total_time_cost + time_turbo
        lstm_time_costs[ix] = lstm_time_cost
        tam_train_time_costs[ix] = tam_train_time_cost
        tam_matching_time_costs[ix] = tam_matching_time_cost
    time_costs = [tam_train_time_costs, tam_matching_time_costs, lstm_time_costs, total_time_costs]
    result = combine_result(graph_matching_results, time_costs, methods, times, attributes, epoch)
    return result

def combine_result(matching_results, time_costs, methods, times, attributes, epoch):
    avg_result = np.array(matching_results).mean(axis=1)
    total_attributes = times + attributes
    result = pd.DataFrame({attr:np.zeros(len(methods)) for attr in total_attributes})
    result.set_axis(methods,axis='index',inplace=True)
    for ix, method in enumerate(methods):
        result.loc[method] = matching_results[ix]
    result['Average'] = avg_result
    time_cost_class = ['TAM Training time', 'TAM Matching time', 'LSTM time', 'Total time']
    for ix, time_class in enumerate(time_cost_class):
        result[time_class] = time_costs[ix]
    print(result)
    print(result['Average'])
    if not os.path.exists(f'../output/{args.dataset}/'):
        os.makedirs(f'../output/{args.dataset}/')
    result.to_csv(f'../output/{args.dataset}/result_{args.iter_num}_{args.intervals}_{args.time_diff_threshold}_{epoch}.csv')
    return result

if __name__ == "__main__":
    seed_everything(args.seed)

    if args.dataset == 'user_1' or args.dataset == 'user_2':
        data_path = '../data/motionsense/' + args.dataset + '/'
        data_files = sorted(os.listdir(data_path))
        data_files = [f for f in data_files if f.count('.csv') > 0]
        attributes = ['userAcceleration.x',
              'userAcceleration.y',
              'userAcceleration.z']
        times = ['time']
        time_graph_start = time.clock()
        graphs = generate_temp_graphs(dataset_name=args.dataset,load_data_func=load_motionsense_data,times = times, attributes=attributes, time_diff_threshold=args.time_diff_threshold, intervals=args.intervals, data_dir=data_path, save_dir='../output/'+args.dataset+'/', load_flag=args.pre_load)
        time_graph_gen = time.clock() - time_graph_start
        print('Graphs Generated')
    else:
        raise ValueError('Wrong Dataset Name')

    path = f'../results/{args.dataset}/'
    if not os.path.exists(path):
        os.makedirs(path)
    results = []
    epochs = 5
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        result = exp_epoch(graphs, times, attributes, args.dataset, time_graph_gen, epoch)
        results.append(result)
    final_result = sum(results) / epochs
    print(final_result)
    final_result.to_csv(f'{path}result_{args.iter_num}_{args.intervals}_{args.time_diff_threshold}.csv')