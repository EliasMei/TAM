import os
import math
import time
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()

EPSILON = 1e-12

class RelationBipartiteGraph:
    def __init__(self, graphs, times, attributes, dataset_name, node_model=None, edge_model=None,random_module=None):
        self.graphs = graphs
        self.times = times
        self.attributes = attributes
        self.dataset_name = dataset_name
        self.random_module = random_module
        self.nodes = times + attributes
        self.node_model = node_model
        self.edge_model = edge_model
        self.edges = [(time_ix, attr_ix) for time_ix, time in enumerate(times) for attr_ix, attr in enumerate(attributes)]
        self.hist = {}
        self.csv_score_dict = self.generate_csv_score_dict()

    def find_possible_match(self):
        nodes_num = len(self.nodes)
        perms = []
        def perm(elem_list, perm_list=[]):
            if len(elem_list) == 0:
                perms.append(perm_list)
                return
            for ix,item in enumerate(elem_list):
                perm(elem_list[:ix] + elem_list[ix+1:], perm_list+[item])
        perm(list(range(nodes_num)))
        # print(f'Num of possible match: {len(perms)}')
        possible_match = []
        for item in perms:
            match = {i:j for i, j in enumerate(item)}
            flag = True
            for ix, _ in enumerate(self.times):
                if match[ix] >= len(self.times):
                    flag = False
                    break
            if flag:
                possible_match.append(match)
        self.random_module.shuffle(possible_match)
        return possible_match

    def standardize_node_score(self, csv_1, csv_2):
        if self.node_model is None:
            return None
        data_file_name_1, data_file_name_2 = csv_1[0][0].file_name, csv_2[0][0].file_name
        node_score_dict = {}
        # print(f'{data_file_name_1} - {data_file_name_2}')
        for n_1 in self.nodes:
            for n_2 in self.nodes:
                score = self.node_model.similarity(data_file_name_1, data_file_name_2, n_1, n_2)
                node_score_dict[(data_file_name_1, data_file_name_2, n_1, n_2)] = score
                node_score_dict[(data_file_name_2, data_file_name_1, n_2, n_1)] = score
                # print(f'{n_1} - {n_2}: {score}')
        scores = np.array([score for _, score in node_score_dict.items()])
        scores_min = scores.min()
        scores_max = scores.max()
        # avoid max == min -> delta = 0
        delta = scores_max - scores_min + EPSILON
        for key, score in node_score_dict.items():
            node_score_dict[key] = (score-scores_min) / delta
        return node_score_dict

    def standardize_edge_score(self, csv_1, csv_2):
        if self.edge_model is None:
            return None
        edge_score_dict = {}
        for edge_1 in self.edges:
            time_1_ix, attr_1_ix = edge_1
            for edge_2 in self.edges:
                time_2_ix, attr_2_ix = edge_2
                graph_1, graph_2 = csv_1[time_1_ix][attr_1_ix], csv_2[time_2_ix][attr_2_ix]
                score = self.edge_model.similarity(graph_1, graph_2)
                edge_score_dict[(graph_1, graph_2)] = score
                edge_score_dict[(graph_2, graph_1)] = score
        scores = np.array([score for pair, score in edge_score_dict.items()])
        scores_min = scores.min()
        scores_max = scores.max()
        # avoid max == min -> delta = 0
        delta = scores_max - scores_min + EPSILON
        for key, score in edge_score_dict.items():
            edge_score_dict[key] = (score-scores_min) / delta
        return edge_score_dict

    def cal_matching_score(self, csv_1, csv_2, match, node_score_dict, edge_score_dict):
        # print(f'match: {match}')
        """
        match is a dict of node pairs: {0:0, 1:3, 2:0}, where key and value is the ix in self.nodes
        """
        score_hist = {'match':match,'node_score':[],'edge_score':[]}
        # cal node score
        data_file_name_1, data_file_name_2 = csv_1[0][0].file_name, csv_2[0][0].file_name
        node_score = 0
        if node_score_dict is not None:
            for n_1_ix, n_2_ix in match.items():
                n_1, n_2 = self.nodes[n_1_ix], self.nodes[n_2_ix]
                score = node_score_dict[(data_file_name_1, data_file_name_2, n_1, n_2)]
                node_score += score
                score_hist['node_score'].append({n_1 + ' - ' + n_2:str(score)})
       # cal edge score
        edge_score = 0
        if edge_score_dict is not None:
            for time_ix, time in enumerate(self.times):
                time_1_ix, time_2_ix = time_ix, match[time_ix]
                if time_2_ix >= len(self.times):
                    # print('time_2_ix >= len(self.times)')
                    continue
                for attr_ix, attr in enumerate(self.attributes):
                    attr_1_ix, attr_2_ix = attr_ix, match[attr_ix+len(self.times)] - len(self.times)
                    if attr_2_ix < 0:
                        # print('attr_2_ix < 0')
                        continue
                    graph_1, graph_2 = csv_1[time_1_ix][attr_1_ix], csv_2[time_2_ix][attr_2_ix]
                    score = edge_score_dict[(graph_1, graph_2)]
                    edge_score += score
                    key = self.times[time_1_ix] + ' - ' + self.attributes[attr_1_ix] + ' ~ ' + self.times[time_2_ix] + ' - ' + self.attributes[attr_2_ix]
                    score_hist['edge_score'].append({key:str(score)})
        score_hist['final_score'] = str(node_score+edge_score)
        return node_score+edge_score, score_hist
        
    def generate_csv_score_dict(self):
        csv_score_dict = {}
        for csv_1_ix, csv_1 in enumerate(self.graphs):
            for ix, csv_2 in enumerate(self.graphs[csv_1_ix+1:]):
                if self.dataset_name == 'excavator':
                    if csv_1[0][0].file_name[:10] == csv_2[0][0].file_name[:10]:
                        continue
                csv_2_ix = csv_1_ix + ix + 1
                data_file_name_1, data_file_name_2 = csv_1[0][0].file_name, csv_2[0][0].file_name
                self.hist[data_file_name_1 + ' - ' + data_file_name_2] = []
                node_score_dict = self.standardize_node_score(csv_1, csv_2)
                edge_score_dict = self.standardize_edge_score(csv_1, csv_2)
                best_match = None
                best_score = -1
                hist_best = None
                # print(f'{data_file_name_1} - {data_file_name_2}')
                possible_match = self.find_possible_match()
                for match in possible_match:
                    score, hist = self.cal_matching_score(csv_1, csv_2, match, node_score_dict, edge_score_dict)
                    # print(f'match: {match} - score: {score}')
                    self.hist[data_file_name_1 + ' - ' + data_file_name_2].append(hist)
                    if score > best_score:
                        best_score = score
                        best_match = match
                        hist_best = hist
                # print(f'Best match: {best_match}')
                if best_match is None:
                    raise ValueError('score is nan')
                csv_score_dict[(csv_1_ix, csv_2_ix)] = best_match
                csv_score_dict[(csv_2_ix, csv_1_ix)] = best_match
        return csv_score_dict
        
    def get_match(self, csv_1_ix, csv_2_ix):
        return self.csv_score_dict[(csv_1_ix, csv_2_ix)]