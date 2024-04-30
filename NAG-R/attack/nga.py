
import sys, os

import numpy as np
import torch
from attack.base_attack import BaseAttack
from torch.nn.parameter import Parameter
from copy import deepcopy
import sys
sys.path.append('...')
import utils
import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn


class NGA(BaseAttack):
    """NGA/FGSM.

    Parameters
    ----------
    model :
        model to attack
    nnodes : int
        number of nodes in the input graph
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
"""


    def  __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False,  decay=0.9, step=0.7, device='cpu'):

        super(NGA, self).__init__(model, nnodes, attack_structure=attack_structure, attack_features=attack_features, decay=0.9, step=0.7, device=device)
        self.decay = decay # cora= 0.6
        self.step = step # cora= 0.95

        assert not self.attack_features, "not support attacking features"

        if self.attack_features:
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

    def attack(self, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, verbose=False, **kwargs):
        
        """Generate perturbations on the input graph.

        Parameters
        ----------
        ori_features : scipy.sparse.csr_matrix
            Original (unperturbed) adjacency matrix
        ori_adj : scipy.sparse.csr_matrix
            Original (unperturbed) node feature matrix
        labels :
            node labels
        idx_train:
            training node indices
        target_node : int
            target node index to be attacked
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        """

        # modified_adj = ori_adj.todense()
        # modified_features = ori_features.todense()
        # modified_adj, modified_features, labels = utils.to_tensor(modified_adj, modified_features, labels, device=self.device)

        modified_adj = ori_adj.clone()
        modified_features = ori_features.clone()

        pseudo_labels = self.surrogate.predict().detach().argmax(1)
        pseudo_labels[idx_train] = labels[idx_train]

        self.surrogate.eval()
        if verbose == True:
            print('number of pertubations: %s' % n_perturbations)

        momentum = torch.zeros_like(modified_adj).detach()

        add_num = 0
        del_num = 0
        changed = []
        modified_adj.requires_grad = True
        change_list = [[], []]
        for i in range(n_perturbations):
            # 梯度的迭代计算
            adj_norm = utils.normalize_adj_tensor(modified_adj)

            adj_norm_nes = adj_norm + self.decay * self.step * momentum
            grad = torch.zeros_like(modified_adj).to(self.device)

            output = self.surrogate(modified_features, adj_norm_nes)
            loss = F.nll_loss(output[[target_node]], pseudo_labels[[target_node]])
            loss.backward(retain_graph=True)
            # loss = loss
            grad = torch.autograd.grad(loss, modified_adj)[0].detach()
            grad = grad[target_node]

            grad_norm = torch.norm(grad, p=1)
            grad = grad / grad_norm
            grad = momentum[target_node] * self.decay + grad
            momentum[target_node] = grad.detach()
            grad_sort = torch.argsort(torch.abs(grad), descending=True)

            for k in grad_sort:
                sign_grad = grad[k].sign()
                if sign_grad > 0 and modified_adj.data[target_node][k] == 0 and k != target_node :  # add

                    modified_adj.data[target_node][k] = 1
                    modified_adj.data[k][target_node] = 1
                    change_list[1].append(int(k))
                    add_num += 1
                    changed.append(k)
                    break

                elif sign_grad < 0 and modified_adj.data[target_node][k] == 1 and k != target_node :  # del
                    modified_adj.data[target_node][k] = 0
                    modified_adj.data[k][target_node] = 0
                    change_list[0].append(int(k))
                    changed.append(k)
                    break

        modified_adj = modified_adj.detach()
        self.check_adj(modified_adj)
        self.modified_adj = modified_adj.detach()
        # self.modified_features = modified_features
        self.add_num = add_num
        self.del_num = del_num
        self.change_list = change_list
        self.pseudo_labels = pseudo_labels
    