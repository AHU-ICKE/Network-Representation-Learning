# -*- coding: UTF-8 -*
from collections import defaultdict
from graph import Graph
import numpy as np
from utils import cmap2C 
import networkx as nx
import sys
#from scipy import sparse
#from scipy.sparse import csc_matrix
import psutil
import gc

def normalized_adj_wgt(graph):
    adj_wgt = graph.adj_wgt
    adj_idx = graph.adj_idx
    norm_wgt = np.zeros(adj_wgt.shape, dtype=np.float32)
    degree = graph.degree
    for i in range(graph.node_num):
        for j in range(adj_idx[i], adj_idx[i + 1]):
            neigh = graph.adj_list[j]
            norm_wgt[j] = adj_wgt[neigh] / np.sqrt(degree[i] * degree[neigh])
    return norm_wgt

def generate_hybrid_matching(ctrl, graph):
    '''Generate matchings using the hybrid method. It changes the cmap in graph object,
    return groups array and coarse_graph_size.'''
    node_num = graph.node_num
    adj_list = graph.adj_list  # big array for neighbors.
    adj_idx = graph.adj_idx  # beginning idx of neighbors.
    adj_wgt = graph.adj_wgt  # weight on edge
    node_wgt = graph.node_wgt  # weight on node
    cmap = graph.cmap
    norm_adj_wgt = normalized_adj_wgt(graph)

    max_node_wgt = ctrl.max_node_wgt

    groups = []  # a list of groups, each group corresponding to one coarse node.
    matched = [False] * node_num

    # SEM: structural equivalence matching.
    jaccard_idx_preprocess(ctrl, graph, matched, groups)  # 拥有相同邻居节点集的节点们粒化为一类
    ctrl.logger.info("# groups have perfect jaccard idx (1.0): %d" % len(groups))
    degree = [adj_idx[i + 1] - adj_idx[i] for i in range(0, node_num)]

    sorted_idx = np.argsort(degree)
    for idx in sorted_idx:
        if matched[idx]:
            continue
        max_idx = idx
        max_wgt = -1
        for j in range(adj_idx[idx], adj_idx[idx + 1]):
            neigh = adj_list[j] #neigh为节点idx的邻居节点
            if neigh == idx:  # KEY: exclude self-loop. Otherwise, mostly matching with itself.
                continue
            curr_wgt = norm_adj_wgt[j]
            if ((not matched[neigh]) and max_wgt < curr_wgt and node_wgt[idx] + node_wgt[neigh] <= max_node_wgt):
                max_idx = neigh
                max_wgt = curr_wgt
        # it might happen that max_idx is idx, which means cannot find a match for the node. 
        matched[idx] = matched[max_idx] = True
        if idx == max_idx:
            groups.append([idx])
        else:
            groups.append([idx, max_idx])
    coarse_graph_size = 0
    for idx in range(len(groups)):
        for ele in groups[idx]:
            cmap[ele] = coarse_graph_size
        coarse_graph_size += 1
    return groups

def jaccard_idx_preprocess(ctrl, graph, matched, groups):
    '''Use hashmap to find out nodes with exactly same neighbors.'''
    neighs2node = defaultdict(list)
    for i in range(graph.node_num):
        neighs = str(sorted(graph.get_neighs(i)))
        neighs2node[neighs].append(i)
    for key in neighs2node.keys():
        g = neighs2node[key]
        if len(g) > 1:
            for node in g:
                matched[node] = True
            groups.append(g)
    return


def create_coarse_graph(stru_comms, attr_comms, attrmat, graph):
        group = 0
        in_comm = {}
        c_set = []
        #attrmat=attrmat.toarray()
       
        for stru_group in range(len(stru_comms)):
            s_set = set(stru_comms[stru_group])
    
            for attr_group in range(len(attr_comms)): 
                a_set = set(attr_comms[attr_group])
                c_set = list(s_set.intersection(a_set))
                if len(c_set) > 1:
                   in_comm[group] = c_set
                   s_set = s_set.difference(in_comm[group])
                   group += 1
            #print("s_set:",list(s_set))
            if len(list(s_set)) > 0:
               in_comm[group] = list(s_set)
               group += 1

        c_mat=[]
        for c_node in in_comm.keys():
            c1_mat=[]
            c3_mat=None
            for ch_node in in_comm[c_node]:
                c1_mat.append(attrmat[ch_node])   
            c2_mat=np.array(c1_mat)
            c3_mat=np.mean(c2_mat,axis=0)  # 求均值
            c_mat.append(c3_mat)
        Attrmat=np.array(c_mat)   
        #Attrmat= csc_matrix(Attrmat) 
        print(psutil.Process().memory_info().rss)
        NewGraph=create_NewGraph(in_comm,graph)
        #print(psutil.Process().memory_info().rss)
        del graph.edge_num,graph.adj_list,graph.adj_idx,graph.adj_wgt,graph.node_wgt,graph.degree,graph.cmap,in_comm
        del stru_comms,attr_comms
        #gc.collect()
        print(psutil.Process().memory_info().rss)
        return Attrmat, NewGraph
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def create_NewGraph(in_comm, graph):
    
    cmap = graph.cmap   
    coarse_graph_size = 0                
    for inc_idx in in_comm.keys():
        for ele in in_comm[inc_idx]:
            cmap[ele] = coarse_graph_size   
        coarse_graph_size += 1 
    newGraph=Graph(coarse_graph_size, graph.edge_num)
    newGraph.finer = graph
    graph.coarser = newGraph
    
    adj_list = graph.adj_list
    adj_idx = graph.adj_idx
    adj_wgt = graph.adj_wgt
    node_wgt = graph.node_wgt
  
    coarse_adj_list = newGraph.adj_list
    
    coarse_adj_idx = newGraph.adj_idx
    coarse_adj_wgt = newGraph.adj_wgt
    coarse_node_wgt = newGraph.node_wgt
    coarse_degree = newGraph.degree
    coarse_adj_idx[0] = 0
    nedges = 0  # number of edges in the coarse graph
    idx=0

    for idx in range(len(in_comm)):  # idx in the graph
        coarse_node_idx = idx  
        neigh_dict = dict()  # coarser graph neighbor node --> its location idx in adj_list. 
        group = in_comm[idx]
        for i in range(len(group)):
            merged_node = group[i]
            if (i == 0):
                coarse_node_wgt[coarse_node_idx] = node_wgt[merged_node]
            else:
                coarse_node_wgt[coarse_node_idx] += node_wgt[merged_node]
            
            istart = adj_idx[merged_node]
            iend = adj_idx[merged_node + 1]   
            for j in range(istart, iend):
               # print("j:",j)  
                k = cmap[adj_list[j]]  # adj_list[j] is the neigh of v; k is the new mapped id of adj_list[j] in coarse graph.
                if k not in neigh_dict:  # add new neigh
                    coarse_adj_list[nedges] = k
                    coarse_adj_wgt[nedges] = adj_wgt[j]
                    neigh_dict[k] = nedges
                    nedges += 1
                else:  # increase weight to the existing neigh
                    coarse_adj_wgt[neigh_dict[k]] += adj_wgt[j]
                # add weights to the degree. For now, we retain the loop. 

                coarse_degree[coarse_node_idx] += adj_wgt[j]
        
        coarse_node_idx += 1
        coarse_adj_idx[coarse_node_idx] = nedges
    
    
    newGraph.edge_num = nedges
    #newGraph.G= graph2nx(newGraph)
    
    newGraph.resize_adj(nedges)
    #newGraph.G=newG
    C = cmap2C(cmap)  # construct the matching matrix.
    graph.C = C
    newGraph.A = C.transpose().dot(graph.A).dot(C)
    return newGraph
