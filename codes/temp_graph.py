import os
import math
import time
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
plt.switch_backend('agg')
tqdm.pandas()

class TempDepNode:
    def __init__(self, value):
        self.value = value
        self.embedding = None

    def set_embedding(self, embedding):
        self.embedding = embedding
    
    def get_embedding(self):
        return self.embedding

class TempDepEdge:
    def __init__(self, from_node, target_node):
        self.from_node = from_node
        self.target_node = target_node
        self.temp_list = list()
        self.embedding = None

    def add_event(self, diff):
        self.temp_list.append(diff)
    
    def get_embedding(self):
        return self.embedding

class TempDepGraph:
    def __init__(self, file_name, dataset_name, time_name, attr_name, data, time_diff_threshold=600, intervals = 20):
        self.file_name = file_name
        self.dataset_name = dataset_name
        self.attr_name = attr_name
        self.time = time_name
        self.data = data
        self.nodes = dict()
        self.edges = dict()
        self.time_diff_threshold = time_diff_threshold
        self.intervals = intervals
        self.edges_freq_weight = None
        self.graph_data = self.discret()
        self.domain = set(self.graph_data[self.attr_name])
        self.build_graph()
        self.set_edge_freq_weight()

    def info(self):
        print(
            f'File name: {self.file_name}, Time: {self.time}, Attribute: {self.attr_name}, Nodes: {len(self.nodes)}, Edges: {len(self.edges)}, Events: {self.event_description()}')

    def discret(self):
        graph_data = copy.deepcopy(self.data)
        graph_data[self.attr_name] = pd.cut(graph_data[self.attr_name], self.intervals)
        graph_data[self.attr_name] = graph_data[self.attr_name].apply(lambda x:x.right)
        return graph_data
    
    def get_min_max(self):
        domain = list(self.domain)
        return min(domain), max(domain)

    def get_temp_list_domain(self):
        temp_list_domain = set()
        for value, edge in self.edges.items():
            temp_list_domain = temp_list_domain.union(set(edge.temp_list))
        return temp_list_domain
    
    def set_edge_freq_weight(self):
        total_weight = sum([len(edge.temp_list) for _, edge in self.edges.items()])
        self.edges_freq_weight = {value:(len(edge.temp_list) / total_weight) for value, edge in self.edges.items()}
        return 

    def set_edge_embedding(self):
        # edge embedding: (emb_i+emb_j) where emb_i is the embedding of from_node and emb_j is the embedding of end_node
        for node_pair, edge in self.edges.items():
            value_i, value_j = node_pair
            vi, vj = self.nodes[value_i], self.nodes[value_j]
            emb_i = vi.get_embedding() 
            emb_j = vj.get_embedding()
            if (emb_i is not None) and (emb_j is not None):
                edge.embedding = emb_i+emb_j

    def aggregation(self):
        embeddigns = np.array([self.edges_freq_weight[value] * edge.embedding for value, edge in self.edges.items()])
        graph_vec = np.mean(embeddigns, axis=0)
        return graph_vec

    def event_description(self):
        event_description = {}
        event_stats = [len(edge.temp_list) for _, edge in self.edges.items()]
        event_description['Num'] = sum(event_stats)
        event_description['Min'] = min(event_stats)
        event_description['Max'] = max(event_stats)
        event_description['Mean'] = sum(event_stats) / len(event_stats)
        return event_description

    def build_graph(self):
        """
        Build graph in which edges are attached between two most recent occurrences of different values
        :return:
        """
        start_time = time.time()

        # init temp node
        for value in self.domain:
            node = TempDepNode(value)
            self.nodes[value] = node

        attr_data = self.graph_data[self.attr_name]
        print(f'{len(attr_data)} records in data')

        # init temp edge
        for source_ix, value_i in tqdm(attr_data.items()):
            visited = set()
            for target_ix, value_j in attr_data[source_ix+1:].items():
                if value_j in visited:
                    continue
                else:
                    visited.add(value_j)
                time_diff = self.graph_data[self.time][target_ix] - \
                    self.graph_data[self.time][source_ix]
                if time_diff > self.time_diff_threshold:
                    break
                if (value_i, value_j) not in self.edges or (value_j, value_i) not in self.edges:
                    self.edges[(value_i, value_j)] = TempDepEdge(value_i, value_j)
                    self.edges[(value_j, value_i)] = TempDepEdge(value_j, value_i)
                self.edges[(value_i, value_j)].add_event(time_diff)
                if value_i != value_j:
                    self.edges[(value_j, value_i)].add_event(time_diff)
        end_time = time.time()
        print(f'{end_time-start_time} seconds for graph building')
