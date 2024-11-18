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

    def hungarian_distance(self, t1, t2, h1, h2):
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


    def forward(self, dense_t1, dense_t2, t1_mask = None, t2_mask = None):



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

            if t1_mask is None:
                t1_features = t1_features[t1_mask[batch_idx]]
                t1_edges = self.produce_reading_order_edges(t1_features.shape[0]).to(self.device)

            if t1_mask is None:
                t2_features = t2_features[t2_mask[batch_idx]]
                t2_edges = self.produce_reading_order_edges(t2_features.shape[0]).to(self.device)

            message_passed_t1 = self.gat(
                x = t1_features,
                edge_index = t1_edges
            )

            message_passed_t2 = self.gat(
                x = t2_features,
                edge_index = t2_edges
            )

            local_distances.append(self.hungarian_distance(t1_features, t2_features,
                                                           message_passed_t1, message_passed_t2
                                                           )
                                   )

            # contextualized_distances.append(self.hungarian_distance(message_passed_t1, message_passed_t2))

        return torch.stack(local_distances)





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


        # self.node_ins_del_cost = nn.Sequential( nn.Linear(args[0], args[0] // 2),
        #                                    nn.ReLU(True),
        #                                    nn.Linear(args[0] //2, 1))
        self.node_del_cost = nn.Sequential( nn.Linear(args[0], args[0] // 2),
                                           nn.ReLU(True),
                                           nn.Linear(args[0] //2, 1))

        self.node_ins_cost = nn.Sequential( nn.Linear(args[0], args[0] // 2),
                                           nn.ReLU(True),
                                           nn.Linear(args[0] //2, 1))

        self.del_cost_per_page = nn.Embedding(5000, 1)
        self.ins_cost_per_page = nn.Embedding(5000, 1)

        self.p = 2

    def cdist(self, set1, set2):
        ''' Pairwise Distance between two matrices
        Input:  x is a Nxd matrix
                y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''
        dist = set1.unsqueeze(1) - set2.unsqueeze(0)
        return dist.abs()


    def hungarian_distance(self, t1, t2, p1, p2):
        # Overrides the hungarian loss to haussdorff
        return self.haussdorf_distance(t1, t2, p1, p2)

    def haussdorf_distance(self, p1, p2, h1, h2, inference=False):
        """
        Compute a modified Hausdorff distance between two sets of points, based on
        both their word embeddings and context embeddings. This implementation
        also supports inference mode where index mappings of the minimum distances are returned.

        Arguments:
        - h1, h2: Context embeddings (e.g., sentences or higher-level text context vectors).
        - p1, p2: Word embeddings (e.g., words or token-level vectors).
        - inference: Boolean flag for whether to return index mappings for distances.

        Returns:
        - If `inference` is False: a scalar distance value.
        - If `inference` is True: distance value and index mappings.
        """

        # Compute the pairwise distance between word embeddings (p1 and p2)
        # using a custom distance function `cdist` (assumed to compute pairwise distances).
        word_dist = self.cdist(p1, p2)

        # Square each element of the pairwise word distances, then sum across the last dimension.
        # This computes a squared Euclidean-like distance across word embeddings.
        word_dist = word_dist.pow(2.).sum(-1)

        # Compute the pairwise distance between context embeddings (h1 and h2), square it,
        # and then sum across the last dimension. This gives a squared distance across contexts.
        context_dist = self.cdist(h1, h2).pow(2.).sum(-1)

        # The final distance matrix is the average of word and context distances.
        dist_matrix = (word_dist + context_dist) / 2

        # Compute insertion/deletion cost for the first context embedding set (h1).
        # The function `node_ins_del_cost` presumably computes the cost of adding/deleting nodes (context vectors).
        # The costs are adjusted by adding 0.5, then taking the absolute value and squeezing any extra dimensions.
        # TODO: I'm now making td a function of the size of the page
        # But this is pure bullshit
        p1len = len(p1)
        p2len = len(p2)
        d1 = self.del_cost_per_page(torch.tensor([i for i in range(max(0, p1len-5), p1len)], dtype=torch.int64, device=self.device)).mean() + self.node_del_cost(p1).abs().squeeze()

        # Similarly, compute insertion/deletion cost for the second context embedding set (h2).
        d2 = self.ins_cost_per_page(torch.tensor([i for i in range(max(0, p2len-5), p2len)], dtype=torch.int64, device=self.device)).mean() + self.node_ins_cost(p2).abs().squeeze()

        # Find the minimum distance for each element in the first set (`p2`/`h2`), across all elements of the second set (`p1`/`h1`).
        # `dist_matrix.min(0)` returns the minimum values along the 0th axis (set2), along with the indices.
        a, indA = dist_matrix.min(0)

        # Update the minimum distance `a` by choosing the smaller of `a` or the corresponding deletion cost `d2`.
        a = torch.min(a, d2)

        # Similarly, find the minimum distance for each element in the second set (`p1`/`h1`) across elements in the first set.
        b, indB = dist_matrix.min(1)

        # Update the minimum distance `b` by choosing the smaller of `b` or the corresponding deletion cost `d1`.
        b = torch.min(b, d1)

        # Calculate the total distance `d` as the sum of the elements in `a` and `b`.
        # The sum of minimum distances from both directions (set1 to set2 and set2 to set1).
        d = a.sum() + b.sum()

        # Normalize the distance by dividing by the number of elements in `a` and `b` (i.e., the number of points in both sets).
        d = d / (a.shape[0] + b.shape[0])

        # If we are not in inference mode, return the computed distance.
        if not inference:
            return d

        # If inference mode is enabled, convert the indices `indA` and `indB` to floating point for further processing.
        indA = indA.to(torch.float32)
        indB = indB.to(torch.float32)

        # Set the indices in `indA` corresponding to points where `a` equals `d2` (i.e., the deletion cost)
        # to infinity (`torch.inf`), meaning these points will be considered unmatched.
        indA[a == d2] = torch.inf

        # Similarly, set the indices in `indB` where `b` equals `d1` to infinity, meaning unmatched.
        indB[b == d1] = torch.inf

        # In inference mode, return the distance, along with the index mappings `indB` and `indA`.
        return d, indB, indA


class SoftHdSimetric(SoftHd):
    '''

        CLASS JUST SO I CAN KEEP OLD STUFF BUT THIS IS A ORDINARIEZ



    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        # self.node_ins_del_cost = nn.Sequential( nn.Linear(args[0], args[0] // 2),
        #                                    nn.ReLU(True),
        #                                    nn.Linear(args[0] //2, 1))
        self.node_ins_del_cost = nn.Sequential( nn.Linear(args[0], args[0] // 2),
                                           nn.ReLU(True),
                                           nn.Linear(args[0] //2, 1))


        self.p = 2

    def haussdorf_distance(self, h1, h2, p1, p2,inference=False):

        word_dist = self.cdist(p1, p2)
        word_dist = word_dist.pow(2.).sum(-1)
        context_dist = self.cdist(h1, h2).pow(2.).sum(-1)
        dist_matrix = (word_dist + context_dist) / 2
        d1 = 0.5 + self.node_ins_del_cost(h1).abs().squeeze()
        d2 = 0.5 + self.node_ins_del_cost(h2).abs().squeeze()
        a, indA = dist_matrix.min(0)
        a = torch.min(a, d2)
        b, indB = dist_matrix.min(1)
        b = torch.min(b, d1)
        d = a.sum() + b.sum()
        d = d / (a.shape[0] + b.shape[0])
        if not inference:
            return d
        indA = indA.to(torch.float32)
        indB = indB.to(torch.float32)
        indA[a == d2] = torch.inf
        indB[b == d1] = torch.inf
        return d, indB, indA