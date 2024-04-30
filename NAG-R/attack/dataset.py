import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
sys.path.append('...')
from utils import get_train_val_test, get_train_val_test_gcn
import zipfile
import json
from collections import defaultdict


class Dataset:
    def __init__(self, root, name, setting='nettack', seed=None, require_mask=False):
        self.name = name.lower()
        self.setting = setting.lower()

        assert self.name in ['cora', 'citeseer', 'cora_ml', 'polblogs',
                             'pubmed', 'reddit', 'acm', 'blogcatalog', 'uai', 'flickr'], \
            'Currently only support cora, citeseer, cora_ml, ' + \
            'polblogs, pubmed, reddit, acm, blogcatalog, flickr'
        assert self.setting in ['gcn', 'nettack', 'prognn'], "Settings should be" + \
                                                             " choosen from ['gcn', 'nettack', 'prognn']"

        self.seed = seed
        self.root = osp.expanduser(osp.normpath(root)) + '/attack/data/dataset/'

        self.data_folder = osp.join(self.root, self.name, self.name)
        self.data_filename = self.data_folder + '.npz'
        self.require_mask = require_mask

        self.require_lcc = False if setting == 'gcn' else True 
        # self.require_lcc = False
        self.adj, self.features, self.labels = self.load_data()

        if setting == 'prognn':
            assert name in ['cora', 'citeseer', 'pubmed', 'cora_ml', 'polblogs'], "ProGNN splits only " + \
                                                                                  "cora, citeseer, pubmed, cora_ml, polblogs"
            self.idx_train, self.idx_val, self.idx_test = self.get_prognn_splits()
        else:
            self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()

    def load_data(self):

        print('Loading {} dataset...'.format(self.name))
        if self.name == 'pubmed':
            return self.load_pubmed()
        if self.name in ['acm', 'blogcatalog', 'uai', 'flickr']:
            return self.load_zip()
        if self.name == 'reddit':
            return self.load_txt()

        adj, features, labels = self.get_adj()
        return adj, features, labels

    def get_train_val_test(self):
        """Get training, validation, test splits according to self.setting (either 'nettack' or 'gcn').
        """
        if self.setting == 'nettack':
            return get_train_val_test(nnodes=self.adj.shape[0], val_size=0.1, test_size=0.8, stratify=self.labels,
                                      seed=self.seed)
        if self.setting == 'gcn':
            return get_train_val_test_gcn(self.labels, seed=self.seed)

    def load_zip(self):
        pass

    def load_txt(self):
        file_dir = osp.join(self.root, self.name, self.name)
        adj = self.read_graph(file_dir + ".edgelist")
        labels = self.read_label(file_dir + ".label")
        features = self.loadDataSet(file_dir + ".features")

        return adj, features, labels

    def graph_to_adj(self, graph, self_loop=False):
        node_num = graph.node_num
        i_arr = []
        j_arr = []
        data_arr = []
        for i in range(0, node_num):
            for neigh_idx in range(graph.adj_idx[i], graph.adj_idx[i + 1]):
                i_arr.append(i)
                j_arr.append(graph.adj_list[neigh_idx])
                data_arr.append(graph.adj_wgt[neigh_idx])
        adj = sp.csr_matrix((data_arr, (i_arr, j_arr)), shape=(node_num, node_num), dtype=np.float32)
        if self_loop:
            adj = adj + sp.eye(adj.shape[0])
        return adj

    def load_pubmed(self):
        dataset = 'pubmed'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = "ind.{}.{}".format(dataset, names[i])
            data_filename = osp.join(self.root, 'pubmed', name)

            with open(data_filename, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        test_idx_file = "ind.{}.test.index".format(dataset)
        test_idx_reorder = parse_index_file(osp.join(self.root, 'pubmed', test_idx_file))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.where(labels)[1]

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)
            adj = adj[lcc][:, lcc]
            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        return adj, features, labels

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)
            adj = adj[lcc][:, lcc]
            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True):

        if not file_name.endswith('.npz'):
            file_name += '.npz'

        with np.load(file_name) as loader:
            # loader = dict(loader)
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                     loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                              loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def get_mask(self):
        pass

    def largest_connected_components(self, adj, n_components=1):
        """Select k largest connected components.
        Parameters
        ----------
        adj : scipy.sparse.csr_matrix
            input adjacency matrix
        n_components : int
            n largest connected components we want to select
        """

        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    def read_label(self, inputFileName):
        f = open(inputFileName, 'r')
        lines = f.readlines()
        f.close()
        N = len(lines)
        y = np.zeros(N, dtype=int)
        for line in lines:
            l = line.strip('\n\r').split(' ')
            y[int(l[0])] = int(l[1])

        return y

    def read_graph(self, inputFileName, is_diercted=False):
        in_file = open(inputFileName)
        neigh_dict = defaultdict(list)
        edge_num = 0
        for line in in_file:
            # print(line)
            eles = line.strip().split()
            n0, n1 = [int(ele) for ele in eles[:2]]
            if len(eles) == 3:  # weighted graph
                wgt = float(eles[2])
                neigh_dict[n0].append((n1, wgt))
                if is_diercted == False and n0 != n1:
                    neigh_dict[n1].append((n0, wgt))
            else:
                neigh_dict[n0].append(n1)
                if is_diercted == False and n0 != n1:
                    neigh_dict[n1].append(n0)
            if is_diercted == False and n0 != n1:
                edge_num += 2
            else:
                edge_num += 1
        in_file.close()
        weighted = (len(eles) == 3)

        node_num = len(neigh_dict)
        graph = Graph(node_num, edge_num, weighted)
        edge_cnt = 0
        graph.adj_idx[0] = 0
        for idx in range(node_num):
            graph.node_wgt[idx] = 1  # default weight to nodes
            for neigh in neigh_dict[idx]:
                if graph.weighted:
                    graph.adj_list[edge_cnt] = neigh[0]
                    graph.adj_wgt[edge_cnt] = neigh[1]
                else:
                    graph.adj_list[edge_cnt] = neigh
                    graph.adj_wgt[edge_cnt] = 1.0
                edge_cnt += 1
            graph.adj_idx[idx + 1] = edge_cnt
        print(" graph.adj_idx", len(graph.adj_idx))
        print(" graph.adj_wgt", len(graph.adj_wgt))

        graph.A = self.graph_to_adj(graph, self_loop=False)
        return graph.A

    def loadDataSet(self, filename):
        fr = open(filename)
        stringArr = []
        line = fr.readline()
        while line:
            items = line.strip().split(' ')
            stringArr.append(items[1:])
            line = fr.readline()
        datArr = [list(map(float, line)) for line in stringArr]
        return sp.csr_matrix(np.array(datArr))


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
