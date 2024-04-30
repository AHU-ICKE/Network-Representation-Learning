import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import operator
import sys
from defense.gcn import GCN
from attack import NGA
from utils import *
from attack import Dataset
from tqdm import tqdm
import argparse
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import math
import time


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'reddit'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--decay', type=float, default=0.2, help='decay')
parser.add_argument('--step', type=float, default=0.2, help='step')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root=os.getcwd(), name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

idx_unlabeled = np.union1d(idx_val, idx_test) # all unlabeled nodes

print("Nodes Num: ", adj.shape[0])
print("Edges Num: ", len(adj.data))
print("Features Num: ",  features.shape[1])
print("Label Max: ", labels.max().item())

dense_adj = adj.todense()
dense_features = features.todense()
tensor_adj, tensor_features, tensor_labels = to_tensor(dense_adj, dense_features, labels, device=device)

# 一阶级邻居
onehops_dict = sparse_mx_to_khopsgraph(adj)


surrogate = GCN(nfeat=tensor_features.shape[1], nclass=tensor_labels.max().item()+1,
                nhid=16, device=device)

surrogate = surrogate.to(device)
surrogate.fit(tensor_features, tensor_adj, tensor_labels, idx_train, idx_val)


def select_nodes(target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                         nhid=16,
                         nclass=labels.max().item() + 1,
                         dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(tensor_features, tensor_adj, tensor_labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0:  # only keep the nodes correctly classifiedrfwhvhu90-
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x: x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 100]]
    low = [x for x, y in sorted_margins[-10:]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 100, replace=False).tolist()
    return high


def single_test(adj, features, target_node, gcn=None):
    if gcn is None:
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)

        gcn = gcn.to(device)

        gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
        gcn.eval()
        output = gcn.predict()
    else:
        output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])

    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

def multi_test_evasion():
    target_gcn = surrogate
    cnt = 0
    degrees = adj.sum(0).A1
    node_list = select_nodes(target_gcn)
    num = len(node_list)

    print('=== [Evasion] Attacking %s nodes respectively ===' % num)

    for target_node in tqdm(node_list):
        n_perturbations = int(degrees[target_node])
        change_list = [[], []]
        model = NGA(surrogate, nnodes=adj.shape[0], decay=args.decay, step=args.step, device=device)
        model = model.to(device)
        model.attack(tensor_features, tensor_adj, tensor_labels, idx_train, target_node, n_perturbations)
        modified_adj = model.modified_adj
        change_list = model.change_list
        add_num = len(change_list[1])
        del_num = len(change_list[0])

        if add_num >= (n_perturbations / 2.):
            add_fin_num = math.ceil(n_perturbations / 2) 
            if n_perturbations % 2 == 0:
                del_fin_num = math.ceil(n_perturbations / 2)  
            else:
                del_fin_num = int(n_perturbations / 2) + 1  

            add_change_num = add_num - add_fin_num
            del_change_num = del_fin_num - del_num

            cosine_arr = {}  
            for j in change_list[1]:
                cosine_arr[j] = torch.cosine_similarity(tensor_features[target_node], tensor_features[j], dim=0)
            cosine_sort = sorted(cosine_arr.items(), key=lambda x: x[1], reverse=True)
            for i in range(add_change_num):
                modified_adj[target_node, cosine_sort[i][0]] = 0.
                modified_adj[cosine_sort[i][0], target_node] = 0.

            cosine_arr_d = {} 
            for k in onehops_dict[target_node]:
                cosine_arr_d[k] = (torch.cosine_similarity(tensor_features[target_node], tensor_features[k], dim=0), int(model.pseudo_labels[k]))
            cosine_sort_d = sorted(cosine_arr_d.items(), key=lambda x: x[1], reverse=True)
            for i in range(del_change_num):
                modified_adj[target_node, cosine_sort_d[i][0]] = 0.
                modified_adj[cosine_sort_d[i][0], target_node] = 0.
        elif del_num > (n_perturbations / 2.) + 1:
            
            del_fin_num = math.ceil(n_perturbations / 2)  
            if n_perturbations % 2 == 0:
                add_fin_num = math.ceil(n_perturbations / 2)  
            else:
                add_fin_num = int(n_perturbations / 2) + 1  

            del_change_num = del_num - del_fin_num
            add_change_num = add_fin_num - add_num

            cosine_arr_d = {}  
            for j in change_list[0]:
                cosine_arr_d[j] = torch.cosine_similarity(tensor_features[target_node], tensor_features[j], dim=0)
            cosine_sort_d = sorted(cosine_arr_d.items(), key=lambda x: x[1]) 
            for i in range(del_change_num):
                modified_adj[target_node, cosine_sort_d[i][0]] = 1.
                modified_adj[cosine_sort_d[i][0], target_node] = 1.

            cosine_arr = {}  
            for k in range(adj.shape[0]):
                if k not in onehops_dict[target_node] and k != target_node:
                    cosine_arr[k] = torch.cosine_similarity(tensor_features[target_node], tensor_features[k], dim=0)
            cosine_sort = sorted(cosine_arr.items(), key=lambda x: x[1])  
            for i in range(add_change_num):
                modified_adj[target_node, cosine_sort[i][0]] = 1.
                modified_adj[cosine_sort[i][0], target_node] = 1.

        acc = single_test(modified_adj, tensor_features, target_node, gcn=target_gcn)
        if acc == 0:
            cnt += 1

    print('misclassification rate : %s' % (cnt / num))
if __name__ == '__main__':
    multi_test_evasion()
