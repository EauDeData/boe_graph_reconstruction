import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import to_dense_batch
import torch.nn as nn


class SoftHungarian(torch.nn.Module):
    '''
     I'm completely stupid.
     Find the most correct implementation here:
     https://github.com/priba/graph_metric.pytorch

     More precisely, I'm replicating this:
     https://github.com/priba/graph_metric.pytorch/blob/master/src/models/distance.py

     Use this implementation at your own risk otherwise.

    ATM Hd stands for Hungarian Distance AHA you didnt expect it.
    I think it makes more sense, just let's try...

    '''
    def __init__(self, emb_dimension, self_attn_layers = 1, cross_attention_layers = 1, td=0.5, ti=0.5, device = 'cuda'):
        super().__init__()

        # For building cliques we will need attention:
        self.device=device
        self.td = td
        self.ti = ti
        self.emb_dim = emb_dimension

        # A single GAT layer
        self.gat = GATConv(emb_dimension, emb_dimension, heads = 1)
        self.to(self.device)


    def cdist(self, set1, set2):
        '''
        Pairwise Distance between two PyTorch matrices
        Input:
            set1 is a Nxd matrix
            set2 is an Mxd matrix
        Output:
            dist is a NxM matrix where dist[i,j] is the squared norm between set1[i,:] and set2[j,:]
        '''
        # Normalize the vectors
        set1_norm = F.normalize(set1, p=2, dim=1)
        set2_norm = F.normalize(set2, p=2, dim=1)

        # Compute cosine similarity matrix
        cosine_similarity_matrix = torch.mm(set1_norm, set2_norm.t())

        return 1 - cosine_similarity_matrix.abs()

    @staticmethod
    def produce_reading_order_edges(num_nodes):
        edges = []
        for i in range(1, num_nodes):
            edges.extend([(i - 1, i), (i, i), (i, i - 1)])
        return torch.tensor(edges, dtype=torch.int32).T

    def hungarian_distance(self, t1, t2):
        '''

        visual_tokens: can be whether post-message passing o pre-message passing
        word_tokens: PRE CONTEXT, otherwise you need context before tagging
        '''
        # Aqui haurem de fer els 2-cliques i 3-cliques amb el mecanisme d'atenció que definim

        cost_matrix = self.cdist(
                                    t1,
                                    t2
                                  )
        row_ind, col_ind = linear_sum_assignment(cost_matrix.clone().detach().cpu().numpy())

        # print(cost_matrix[row_ind, col_ind].mean())
        return cost_matrix[row_ind, col_ind].mean()


    def forward(self, dense_t1, dense_t2):



        t1bs, t2bs = dense_t1.shape[1], dense_t2.shape[1]

        local_distances = [ ]
        contextualized_distances = []

        t1_edges = self.produce_reading_order_edges(t1bs).to(self.device)
        t2_edges = self.produce_reading_order_edges(t2bs).to(self.device)

        assert dense_t2.shape[0] == dense_t1.shape[0], ('Care with this batch,'
                                                        'query and document batches do not match...')
        for batch_idx in range(dense_t2.shape[0]):

            t1_features = dense_t1[batch_idx]
            t2_features = dense_t2[batch_idx]

            local_distances.append(self.hungarian_distance(t1_features, t2_features))

            message_passed_t1 = self.gat(
                x = t1_features,
                edge_index = t1_edges
            )

            message_passed_t2 = self.gat(
                x = t2_features,
                edge_index = t2_edges
            )


            contextualized_distances.append(self.hungarian_distance(message_passed_t1, message_passed_t2))

        return (
            torch.stack(local_distances),
            torch.stack(contextualized_distances)
        )


class SoftHd(SoftHungarian):
    '''
     I'm completely stupid.
     Find the most correct implementation here:
     https://github.com/priba/graph_metric.pytorch

     More precisely, I'm replicating this:
     https://github.com/priba/graph_metric.pytorch/blob/master/src/models/distance.py

     Use this implementation at your own risk otherwise.



    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.node_ins_del_cost = nn.Sequential( nn.Linear(args[0], args[0] // 2),
                                           nn.ReLU(True),
                                           nn.Linear(args[0] //2, 1))

        self.p = 2

    def cdist(self, set1, set2):
        ''' Pairwise Distance between two matrices
        Input:  x is a Nxd matrix
                y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''
        dist = set1.unsqueeze(1) - set2.unsqueeze(0)
        return dist.abs().pow(2.).sum(-1)


    def hungarian_distance(self, t1, t2):
        # Overrides the hungarian loss to haussdorff
        return self.haussdorf_distance(t1, t2)


    def haussdorf_distance(self, t1, t2):

        dist_matrix = self.cdist(t1, t2)

        d1 = 0.5 + self.node_ins_del_cost(t1).abs().squeeze()
        d2 = 0.5 + self.node_ins_del_cost(t2).abs().squeeze()

        # \sum_{a\in set1} \inf_{b_\in set2} d(a,b)
        a, indA = dist_matrix.min(0)
        a = torch.min(a, d2)

        # \sum_{b\in set2} \inf_{a_\in set1} d(a,b)
        b, indB = dist_matrix.min(1)
        b = torch.min(b, d1)

        #d = a.mean() + b.mean()
        d = a.sum() + b.sum()
        # d = d/(d1.sum() + d2.sum()) # (a.shape[0] + b.shape[0])

        d = d/(a.shape[0] + b.shape[0])

        return d