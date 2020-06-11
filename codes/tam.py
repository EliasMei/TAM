import time
import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
from temp_graph import *
from data_processor import *
from util import *
from graph_matching import *
from train import *

class TAM:
    def __init__(self, graphs, pre_trained=False, path='../output/model/graph_ae.pt', max_epoch=150, max_len=20, batch_size=128, node_emb_size=50, iter_num=2, stored=False):

        data_loader = get_graph_dataloader(graphs=graphs, max_len=max_len, batch_size=batch_size, shuffle=False)
        init_node_embedding = get_init_node_embedding(graphs, node_emb_size=node_emb_size)
        train_time_start = time.clock()
        if pre_trained:
            model = Graph_AutoEncoder(hidden_size=100, init_node_embedding=init_node_embedding)
            model.load_state_dict(torch.load(path)['state_dict'])
        else:
            while(True):
                model = Graph_AutoEncoder(hidden_size=100,
                init_node_embedding=init_node_embedding)
                model, loss_list = train(model, data_loader=data_loader,
                    path = path,
                    init_node_embedding=init_node_embedding, 
                    hidden_size=100, 
                    max_epoch=max_epoch
                )
                if model is not None:
                    break
        self.train_time = time.clock() - train_time_start
        graphs = embed_graph(graphs, model)
        self.graphs = graphs
        self.dataset_name = self.graphs[0].dataset_name
        self.stored = stored
        self.iter_num = iter_num
        matching_time_start = time.clock()
        self.graph_matching_score_dict = self.generate_matching()
        self.matching_time = time.clock() - matching_time_start

    def generate_matching(self):
        graph_matching_score_dict = dict()
        if self.stored:
            with open('../output/tam_matching_result.pkl', 'rb') as f:
                graph_matching_score_dict = pickle.load(f)
                for k,v in graph_matching_score_dict.items():
                    print(f'{k[0].file_name} - {k[0].time} - {k[0].attr_name} ~ {k[1].file_name} - {k[1].time} - {k[1].attr_name}')
                    print(f'{k[0]} - {k[1]}')
            return graph_matching_score_dict
        start_time = time.time()
        for ix_1, graph_1 in tqdm(enumerate(self.graphs)):
            for graph_2 in self.graphs[ix_1+1:]:
                if graph_1.file_name == graph_2.file_name:
                    continue
                if self.dataset_name == 'excavator':
                    if graph_1.file_name[:10] == graph_2.file_name[:10]:
                        continue
                hm = HeurisiticMatching(graph_1, graph_2, iter_num=self.iter_num)
                matching, score = hm.match()
                score = np.exp(-np.sqrt(score))
                graph_matching_score_dict[(graph_1, graph_2)] = score
                graph_matching_score_dict[(graph_2, graph_1)] = score
        end_time = time.time()
        print(f'Time cost for Graph Matching: {end_time-start_time}')
        with open('../output/tam_matching_result.pkl', 'wb') as f:
                pickle.dump(graph_matching_score_dict, f)
        return graph_matching_score_dict

    def similarity(self, graph_1, graph_2):
        return self.graph_matching_score_dict[(graph_1, graph_2)]