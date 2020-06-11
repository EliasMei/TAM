import time
import pickle
import numpy as np
from tqdm import tqdm
from temp_graph import *

import numpy as np


class FOD:
    """
    Baseline - FOD
    Calculate the First-Order dissimilarity (continuous and discret)
    
    Args:
        all_graphs: a 3-layer list of graphs: [csv[time[attr.graph]]]
            e.x.:
            [[csv1[time_1[attr_1.graph, attr_2.graph, ...], time_2[...]], [csv1[time_1[attr_1.graph, attr_2.graph, ...], time_2[...], [...]]
    """
    def __init__(self, graphs, times, attributes):
        self.times = times
        self.attributes = attributes
        self.graphs = graphs
        self.pmfs = self.generate_pmfs()
        self.similarity_dict = self.generate_similarity_dict()
    
    def cal_pmf_score(self, pmf_1, pmf_2):
        max_len = max(len(pmf_1), len(pmf_2))
        pmf_1 = np.pad(pmf_1, ((0, max_len-len(pmf_1))), 'constant')
        pmf_2 = np.pad(pmf_2, ((0, max_len-len(pmf_2))), 'constant')
        return np.sum(np.square(pmf_1-pmf_2))

    def generate_pmf(self, data):
        """generate probability mass function for series data
        """
        pmf = np.array(data.value_counts()/len(data))
        return pmf

    def generate_pmfs(self):
        pmfs = {}
        for csv_ix, csv_graphs in enumerate(self.graphs):
            data_file_name = csv_graphs[0][0].file_name
            pmfs[data_file_name] = {}
            for time_ix, time_graphs in enumerate(csv_graphs):
                time = time_graphs[0].time
                pmfs[data_file_name][time] = self.generate_pmf(time_graphs[0].data[time])
                pmf = np.array(time_graphs[0].data[time].value_counts()/len(time_graphs[0].data[time]))
                for attr_ix, graph in enumerate(time_graphs):
                    attr_name = graph.attr_name
                    pmfs[data_file_name][attr_name] = self.generate_pmf(graph.data[attr_name])
        return pmfs
    
    def generate_similarity_dict(self):
        similarity_dict = {}
        total_attributes = self.times + self.attributes
        for csv_ix_1, csv_graphs_1 in enumerate(self.graphs):
            data_file_name_1 = csv_graphs_1[0][0].file_name
            for csv_ix_2, csv_graphs_2 in enumerate(self.graphs[csv_ix_1+1:]):
                data_file_name_2 = csv_graphs_2[0][0].file_name
                for attr_1 in total_attributes:
                    key_1 = data_file_name_1 + attr_1
                    pmf_1 = self.pmfs[data_file_name_1][attr_1]
                    for attr_2 in total_attributes:
                        key_2 = data_file_name_2 + attr_2
                        pmf_2 = self.pmfs[data_file_name_2][attr_2]
                        similarity = np.exp(-np.sqrt(self.cal_pmf_score(pmf_1, pmf_2)))
                        similarity_dict[(key_1, key_2)] = similarity
                        similarity_dict[(key_2, key_1)] = similarity
        return similarity_dict

    def similarity(self, data_file_name_1, data_file_name_2, attr_1, attr_2):
        key_1 = data_file_name_1 + attr_1
        key_2 = data_file_name_2 + attr_2
        return self.similarity_dict[(key_1, key_2)]
