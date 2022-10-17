#!/usr/bin/env python
# -*- coding: UTF-8 -*
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from coarsen import generate_hybrid_matching
from embed import multilevel_embed
from utils import read_graph, setup_custom_logger,loadDataSet,normalized,Control
from classification import node_classification_F1, read_label
import logging
import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans, KMeans
import psutil
import gc
import os
from sklearn import metrics

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--data', required=False, default='/cora',
                        help='Input graph file')
    parser.add_argument('--embed-dim', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--coarsen-level', default=2, type=int,
                        help='MAX number of levels of coarsening.')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of workers.')
    parser.add_argument('--lambd', default=0.63, type=float,
                        help='the regularization parameter.')  # usually in the range [0, 1]
    parser.add_argument('--rho', default=8, type=float,
                        help='the penalty parameter.')  # usually in the range [0, 1]
    args = parser.parse_args()
    return args


def set_control_params(ctrl, args, graph):
    ctrl.coarsen_level = args.coarsen_level
    ctrl.embed_dim = args.embed_dim
    ctrl.data = args.data
    ctrl.rho=args.rho
    ctrl.lambd=args.lambd
    ctrl.workers = args.workers
    ctrl.max_node_wgt = int((5.0 * graph.node_num) / ctrl.coarsen_to)
    ctrl.logger = setup_custom_logger('HAANE')
    if ctrl.debug_mode:
        ctrl.logger.setLevel(logging.DEBUG)
    else:
        ctrl.logger.setLevel(logging.INFO)
    ctrl.logger.info(args)


def read_data(ctrl, args):
    prefix = "./dataset" + args.data + args.data
    # input_graph_path = "../dataset/cora/cora.edgelist"  # 输入的拓扑结构
    # input_attr_path = "/Share/home/E19201088/CANE/HAANE/dataset/yelp/yelp_feats.npy"  # yelp属性
    # input_label_path = "/Share/home/E19201088/CANE/HAANE/dataset/yelp/yelp_class_map.json" # yelp标签

    input_graph_path = prefix + ".edgelist"
    input_attr_path = prefix + ".features"
    input_label_path= prefix + ".label"
    label = read_label(input_label_path)
    ctrl.k = len(set(label))
    dataMat = loadDataSet(input_attr_path)
    dataMat = normalized(dataMat, per_feature=False)  # 归一化 l2类型   节点属性（2708,1433）
    del label
    gc.collect()    # 清内存
    graph = read_graph(ctrl, input_graph_path)   #utils.py  读结构信息
    return graph, dataMat, input_label_path


if __name__ == "__main__":
    seed = 123
    np.random.seed(seed)
    #tf.random.set_seed()

    ctrl = Control()
    args = parse_args()

    # Read input graph
    #获得结构信息及属性信息：
        # 结构：节点数，边数; 每个节点的邻居节点情况
        # 属性：所有节点的属性信息
    graph, lowDAttrMat, input_label_path= read_data(ctrl, args)
    set_control_params(ctrl, args, graph)

    # Coarsen method
    match_method = generate_hybrid_matching

    # Generate embeddings
    embeddings = multilevel_embed(ctrl, graph, match_method=match_method, AttrMat=lowDAttrMat)
    print(psutil.Process().memory_info().rss)
    del lowDAttrMat
    gc.collect()
    print(psutil.Process().memory_info().rss)
        
    y = read_label(input_label_path)
    # mbk = KMeans(init='k-means++', n_clusters=ctrl.k)
    # mbk.fit(embeddings)
    # labels = mbk.labels_
    # print(labels)
    # NMI = metrics.normalized_mutual_info_score(labels,y)
    # print("NMI:", NMI)
    for test_rio in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
        print("train_rio",1-test_rio)
        node_classification_F1(embeddings, y, test_rio)

