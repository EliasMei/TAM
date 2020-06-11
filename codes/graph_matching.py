import time
import copy
import torch
import numpy as np
from temp_graph import *
from data_processor import *
from util import *
from train import *


class KM:
    """find the match based on KM algorithm

    See nodes in two given graphs as nodes in the bipartite graph. KM algorithm could find a match which maximize the node matching score

    Args:
        graph_1: the smaller graph (with less nodes)
        graph_2: the bigger graph (with more nodes)
    """

    def __init__(self, graph_1, graph_2):
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.nodes_1 = [node for _, node in graph_1.nodes.items()]
        self.nodes_2 = [node for _, node in graph_2.nodes.items()]
        self.nodes_num = len(self.nodes_2)
        self.matching_matrix = self.init_matching_matrix()

    def init_matching_matrix(self):
        """initialize the matching matrix

        Build a adj matrix to denote the graph. If the sizes of V1 and V2 are different, we need to add some virtual nodes and virtual edges.
        The value in the matrix is exp(-Euclidean distance(vi, vj)). For virtual edges, the weights are 0.
        """
        matching_matrix = np.zeros((self.nodes_num, self.nodes_num))
        for i in range(self.nodes_num):
            for j in range(self.nodes_num):
                if i < len(self.nodes_1):
                    vi, vj = self.nodes_1[i], self.nodes_2[j]
                    matching_matrix[i][j] = np.exp(
                        -np.sqrt(distance_temp(vi, vj)))
                else:
                    matching_matrix[i][j] = 0
        return matching_matrix.astype(np.float32)

    def init_matching(self):
        self.label_X = np.max(self.matching_matrix, axis=1)
        self.label_Y = np.zeros(self.matching_matrix.shape[1])
        self.match_X = np.ones_like(self.label_X) * (-1)
        self.match_X = self.match_X.astype(np.int32)
        self.match_Y = np.ones_like(self.label_Y) * (-1)
        self.match_Y = self.match_Y.astype(np.int32)
        self.feasible_labeling_X = {u: [v] for u, v in enumerate(
            np.argmax(self.matching_matrix, axis=1))}
        self.feasible_labeling_Y = {}
        for u, v in enumerate(np.argmax(self.matching_matrix, axis=1)):
            if v not in self.feasible_labeling_Y:
                self.feasible_labeling_Y[v] = [u]
            else:
                self.feasible_labeling_Y[v].append(u)

        self.match_X = np.argmax(self.matching_matrix, axis=1)
        for ix, y in enumerate(self.match_X):
            if self.match_Y[y] == -1:
                self.match_Y[y] = ix
            else:
                self.match_X[ix] = -1

    def check_matching(self):
        if np.count_nonzero(self.match_X == -1) == 0 and np.count_nonzero(self.match_Y == -1) == 0:
            return True
        if np.count_nonzero(self.match_X != -1) != np.count_nonzero(self.match_Y != -1) == 0:
            raise ValueError('Wrong Matching')
        return False

    def update_labeling(self, S, T):
        # update feasible lableing
        candidate_labeling = {(x, y): (self.label_X[x] + self.label_Y[y] - self.matching_matrix[x][y])
                              for x in S for y in set(range(self.nodes_num)).difference(T)}
        sorted_candidate_labeling = sorted(
            candidate_labeling.items(), key=lambda x: x[1])
        new_labeling, alpha_l = sorted_candidate_labeling[0]
        self.feasible_labeling_X[new_labeling[0]].append(new_labeling[1])
        if new_labeling[1] not in self.feasible_labeling_Y:
            self.feasible_labeling_Y[new_labeling[1]] = [new_labeling[0]]
        else:
            self.feasible_labeling_Y[new_labeling[1]].append(new_labeling[0])

        # update labels
        for v in S:
            self.label_X[v] = self.label_X[v] - alpha_l
        for v in T:
            self.label_Y[v] = self.label_X[v] + alpha_l

    def find_augment_path(self, head, end, path=[], visit={'x': [], 'y': []}, start_x=True):
        """Find augment path (DFS)
        Args:
            head: source node
            end: end node
            path: record previous path
            visit: record nodes which were visited
            start_x: if head is from X, start_x=True. Otherwise, start_x = False
        """
        path.append(head)
        if start_x:
            if end in self.feasible_labeling_X[head]:
                path.append(end)
                return path, True
            else:
                visit['x'].append(head)
                for next_node in self.feasible_labeling_X[head]:
                    if next_node in visit['y']:
                        continue
                    path, flag = self.find_augment_path(
                        next_node, end, path, visit, start_x=False)
                    if not flag:
                        path.pop()
                    else:
                        return path, True
            return path, False
        else:
            visit['y'].append(head)
            for next_node in self.feasible_labeling_Y[head]:
                if next_node in visit['x']:
                    continue
                path, flag = self.find_augment_path(
                    next_node, end, path, visit, start_x=True)
                if not flag:
                    path.pop()
                else:
                    return path, True
        return path, False

    def match(self):
        start_time = time.time()
        self.init_matching()
        while(not self.check_matching()):
            S, T = set(), set()
            u = np.where(self.match_X == -1)[0][0]
            S.add(u)
            N_s = {v for u in S for v in self.feasible_labeling_X[u]}
            while(True):
                if N_s == T:
                    self.update_labeling(S, T)
                    N_s = {v for u in S for v in self.feasible_labeling_X[u]}
                if N_s != T:
                    y = list(N_s.difference(T))[0]
                    if self.match_Y[y] == -1:
                        aug_path, flag = self.find_augment_path(
                            u, y, path=[], visit={'x': [], 'y': []})
                        if not flag:
                            raise ValueError('No Argumen Path')
                        for ix, node_ix in enumerate(aug_path):
                            if ix % 2 == 0:
                                self.match_X[node_ix] = aug_path[ix+1]
                            else:
                                self.match_Y[node_ix] = aug_path[ix-1]
                        break
                    else:
                        S.add(self.match_Y[y])
                        T.add(y)
        matching = {i: self.match_X[i]
                    for i in range(len(self.nodes_1))}
        end_time = time.time()
        # print(f'KM matching: {matching}')
        # print(f'KM finished. Time: {round(end_time-start_time,2)}')
        return matching


class HeurisiticMatching:
    """find the match by updating KM algorithm

    See nodes in two given graphs as nodes in the bipartite graph. KM algorithm could find a initial matching which maximize the node matching score. Then, the algorithm update the initial matching to find a new matching with higher matching score.

    Args:
        graph_1: the smaller graph (with less nodes)
        graph_2: the bigger graph (with more nodes)
    """

    def __init__(self, graph_1, graph_2, iter_num=2, eta=1):
        if len(graph_1.nodes) > len(graph_2.nodes):
            graph = graph_1
            graph_1 = graph_2
            graph_2 = graph
        km = KM(graph_1, graph_2)
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        self.nodes_1 = [node for _, node in graph_1.nodes.items()]
        self.nodes_2 = [node for _, node in graph_2.nodes.items()]
        self.edges_1 = [edge for _, edge in graph_1.edges.items()]
        self.edges_2 = [edge for _, edge in graph_2.edges.items()]
        self.node_dissimilarity_score_dict = self.generate_node_dissimilarity_score_dict()
        self.edge_dissimilarity_score_dict = self.generate_edge_dissimilarity_score_dict()
        self.matching = km.match()
        self.iter_num = iter_num
        self.eta = eta

    def generate_node_dissimilarity_score_dict(self):
        node_dissimilarity_score_dict = {}
        for ix, vi in enumerate(self.nodes_1):
            for vj in self.nodes_2:
                score = distance_temp(vi, vj)
                node_dissimilarity_score_dict[(vi, vj)] = score
                node_dissimilarity_score_dict[(vj, vi)] = score
        return node_dissimilarity_score_dict

    def generate_edge_dissimilarity_score_dict(self):
        edge_dissimilarity_score_dict = {}
        for vi in self.nodes_1:
            for vj in self.nodes_1:
                e_ij = (vi, vj)
                e_ji = (vj, vi)
                for va in self.nodes_2:
                    for vb in self.nodes_2:
                        m_eij = (va, vb)
                        m_eji = (vb, va)
                        score = distance_temp(e_ij, m_eij)
                        edge_dissimilarity_score_dict[(e_ij, m_eij)] = score
                        edge_dissimilarity_score_dict[(e_ij, m_eji)] = score
                        edge_dissimilarity_score_dict[(e_ji, m_eij)] = score
                        edge_dissimilarity_score_dict[(e_ji, m_eji)] = score
        return edge_dissimilarity_score_dict

    def cal_matching_score(self, matching):
        """calculate matching score for temnral graph

        matching score = node_score + edge_score.

        Args:
            matching: a dict whose key is the ix of vi in nodes_1 and value is its matching score

        """
        node_score_dic = {}
        node_score = 0
        edge_score = 0
        for vi_ix, vi in enumerate(self.nodes_1):
            m_vi = self.nodes_2[matching[vi_ix]]
            for vj_ix, vj in enumerate(self.nodes_1):
                # cal_node_score
                m_vj = self.nodes_2[matching[vj_ix]]
                freq_weight = 0
                if (vi.value, vj.value) in self.graph_1.edges_freq_weight:
                    freq_weight += self.graph_1.edges_freq_weight[(
                        vi.value, vj.value)]
                if (m_vi.value, m_vj.value) in self.graph_2.edges_freq_weight:
                    freq_weight += self.graph_2.edges_freq_weight[(
                        m_vi.value, m_vj.value)] 
                dis_i = freq_weight * self.node_dissimilarity_score_dict[(vi, m_vi)] / 2
                dis_j = freq_weight * self.node_dissimilarity_score_dict[(vj, m_vj)] / 2
                node_score += (dis_i + dis_j)
                if vi_ix not in node_score_dic:
                    node_score_dic[vi_ix] = dis_i
                else:
                    node_score_dic[vi_ix] += dis_i
                if vj_ix not in node_score_dic:
                    node_score_dic[vj_ix] = dis_j
                else:
                    node_score_dic[vj_ix] += dis_j
                e_ij = (vi, vj)
                m_eij = (m_vi, m_vj)
                dis = self.edge_dissimilarity_score_dict[(e_ij, m_eij)]
                edge_score += freq_weight * dis
        return node_score, edge_score, node_score_dic

    def update_matching(self, node_score_dic, node_score_curr, edge_score_curr):
        node_score_new, edge_score_new = None, None
        matching_curr = self.matching
        # sorted_matched_pairs: [((vi,vj),score), (...), ...]
        sorted_matched_pairs = sorted(
            node_score_dic.items(), key=lambda x: x[1], reverse=True)
        matched_nodes_ixs = {self.matching[vi_ix]
                             for vi_ix, vi in enumerate(self.nodes_1)}
        unmatched_nodes_ixs = {vj_ix for vj_ix, _ in enumerate(
            self.nodes_2)}.difference(matched_nodes_ixs)

        matching_best = matching_curr
        node_score_best, edge_score_best = node_score_curr, edge_score_curr
        score_best = node_score_best + edge_score_best
        node_score_dic_best = node_score_dic

        # find a candidate match from unmatched nodes in graph 2
        for ix_1, temp_1 in enumerate(sorted_matched_pairs):
            vi_ix = temp_1[0]
            m_vi_ix = matching_best[vi_ix]
            for v_un_ix in unmatched_nodes_ixs:
                matching_new = copy.deepcopy(matching_best)
                node_score_new, edge_score_new, node_score_dic_new = self.cal_matching_score(
                    matching_new)
                score_new = node_score_new + edge_score_new
                if score_new < score_best:
                    matching_best = matching_new
                    score_best = score_new
                    node_score_best = node_score_new
                    edge_score_best = edge_score_new
                    print(node_score_best, edge_score_best)
                    node_score_dic_best = node_score_dic_new

        # find a new match by switching exist matching
        for ix_1, temp_1 in enumerate(sorted_matched_pairs):
            vi_ix = temp_1[0]
            for temp_2 in sorted_matched_pairs[ix_1+1:]:
                m_vi_ix = matching_best[vi_ix]
                cand_vi_ix = temp_2[0]
                cand_m_vi_ix = matching_best[cand_vi_ix]
                matching_new = copy.deepcopy(matching_best)
                matching_new[vi_ix] = cand_m_vi_ix
                matching_new[cand_vi_ix] = m_vi_ix
                node_score_new, edge_score_new, node_score_dic_new = self.cal_matching_score(
                    matching_new)
                score_new = node_score_new + edge_score_new
                if score_new < score_best:
                    matching_best = matching_new
                    score_best = score_new
                    node_score_best = node_score_new
                    edge_score_best = edge_score_new
                    node_score_dic_best = node_score_dic_new
        return matching_best, node_score_dic_best, node_score_best, edge_score_best

    def match(self):
        node_score_curr, edge_score_curr, node_score_dic = self.cal_matching_score(self.matching)
        score_curr = node_score_curr + edge_score_curr
        # print(f'KM Score: {score_curr}')
        count = 0
        for i in range(self.iter_num):
            start_time = time.time()
            score_prev = score_curr
            matching_curr, node_score_dic, node_score_curr, edge_score_curr = self.update_matching(
                node_score_dic, node_score_curr, edge_score_curr)
            score_curr = node_score_curr + edge_score_curr
            end_time = time.time()
            self.matching = matching_curr
            count += 1
            if score_curr > score_prev:
                raise ValueError('Update matching wrong!')
        return self.matching, score_curr
