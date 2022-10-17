# -*- coding: UTF-8 -*
import time
import tensorflow as tf
from coarsen import create_coarse_graph
from utils import normalized, graph_to_adj
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import sys
from scipy import sparse
from scipy.sparse import csc_matrix
import scipy.io as sio
from AANE_fun import AANE
import psutil
import gc
import scipy.sparse as sp
from scipy import linalg
from scipy.special import iv
from utils import  normalized
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd

def print_coarsen_info(ctrl, g):
    cnt = 0
    while g is not None:
        ctrl.logger.info("Level " + str(cnt) + " --- # nodes: " + str(g.node_num))
        g = g.coarser
        cnt += 1

def multilevel_embed(ctrl, graph, match_method,  AttrMat):
    '''This method defines the multilevel embedding method.'''

    start11 = time.time()

    # Step-1: Graph Coarsening.
    original_graph = graph
    stru_comms = {}  # 需定义一个属性一个结构一个最终的
    comms = {}
    graph.Attr = AttrMat
    print("graph.Attr", graph.Attr.shape)

    for i in range(ctrl.coarsen_level):
        time_start = time.time()
        # 将节点按属性进行聚类，即属性社团划分
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=ctrl.k, batch_size=125, n_init=10,
                              max_no_improvement=10, verbose=0, reassignment_ratio=0.001)
        mbk.fit(AttrMat)
        labels = mbk.labels_
        attr_comms = [[] for i in range(ctrl.k)]
        ii = 0
        for item in labels:
            attr_comms[item].append(ii)
            ii += 1
        time_end = time.time()       
        print('kmeans totally cost', time_end - time_start)
        #将节点按结构进行社团划分
        stru_comms[i] = match_method(ctrl, graph)  # MILE粗化
        AttrMat, coarse_graph = create_coarse_graph(stru_comms=stru_comms[i], attr_comms=attr_comms, attrmat=AttrMat, graph=graph)
        
        coarse_graph.Attr = AttrMat
        print(" coarse_graph.Attr.shape", coarse_graph.Attr.shape)
        graph.Attr = csc_matrix(graph.Attr)
        graph.A = csc_matrix(graph.A)
        gc.collect()
        graph = coarse_graph
        if graph.node_num <= ctrl.embed_dim:
            ctrl.logger.error("Error: coarsened graph contains less than embed_dim nodes.")
            exit(0)
 
    del AttrMat
    gc.collect()

    print_coarsen_info(ctrl, original_graph)
    del original_graph

    ## Step-2 : Base Embedding
    A = graph.Attr
    G = normalized(graph.A.toarray())
    print(A.shape)
    print(G.shape)
    print("#############:",type(G))
    del graph.Attr
    del graph.A  
    gc.collect() 
    issvd = True
   
    #features_matrix = pre_factorization(ctrl,G, G)
    print("Accelerated Attributed Network Embedding (AANE):")
    start_time = time.time()
    initial_embed = np.concatenate((G, A), axis=1)
    embeddings = AANE(G, csc_matrix(A), ctrl.embed_dim, initial_embed, issvd, ctrl.lambd, ctrl.rho).function()
    #print("$$$$$",psutil.Process().memory_info().rss)
    #del A,G
    #gc.collect()
    
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
   
    issvd=False
    while graph.finer is not None:
          graph = graph.finer
          initial_embed = graph.C.dot(embeddings)
          print(psutil.Process().memory_info().rss)
          del graph.coarser
          A=graph.Attr
          G=graph.A
                    
          del graph.Attr
          del graph.A
          gc.collect()
          print(psutil.Process().memory_info().rss)
          print("Accelerated Attributed Network Embedding (AANE),")
          start_time = time.time()
          
          embeddings = AANE(G, A, ctrl.embed_dim,initial_embed,issvd,ctrl.lambd, ctrl.rho).function()
          print(psutil.Process().memory_info().rss)        
          del A,G
          gc.collect()
          print(psutil.Process().memory_info().rss)         
          print("time elapsed: {:.2f}s".format(time.time() - start_time))
                     
    end = time.time()
    print("time:",end-start11)
    return embeddings
