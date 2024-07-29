import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class SoftHd(torch.nn.Module):
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
    def __init__(self, emb_dimension, td=0.5, ti=0.5, device = 'cuda'):
        super().__init__()

        # For building cliques we will need attention:
        self.attention_query = torch.nn.Sequential(torch.nn.ReLU(),
                                                   torch.nn.Linear(emb_dimension, emb_dimension)
                                                   )
        self.attention_key = torch.nn.Sequential(torch.nn.ReLU(),
                                                   torch.nn.Linear(emb_dimension, emb_dimension)
                                                   )
        self.sqrt_dim = emb_dimension ** .5
        self.device=device
        self.td = td
        self.ti = ti
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
    def _cdist(self, set1, set2):
        ''' Pairwise Distance between two matrices
        Input:  x is a Nxd matrix
                y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        '''
        dist = set1.unsqueeze(1) - set2.unsqueeze(0)
        return dist.abs()

    def create_2clique(self, nodes):
        '''

        Given a set of nodes return an expanded version consisting on the nodes itself and its potential 2-cliques

        '''
        query = self.attention_query(nodes)
        key = self.attention_key(nodes)
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        pass

    def hungarian_distance(self, visual_tokens, word_tokens):
        '''

        visual_tokens: can be whether post-message passing o pre-message passing
        word_tokens: PRE CONTEXT, otherwise you need context before tagging
        '''
        # Aqui haurem de fer els 2-cliques i 3-cliques amb el mecanisme d'atenció que definim
        cost_matrix = self.cdist(visual_tokens, word_tokens)
        row_ind, col_ind = linear_sum_assignment(cost_matrix.clone().detach().cpu().numpy())
        return cost_matrix[row_ind, col_ind].mean()

    def forward(self,  ocr_features, query_features, ocr_idxs, query_idxs, images, dates):

        d = []
        for batch, (visual_idxs, text_idxs) in enumerate(zip(ocr_idxs, query_idxs)):
            # TODO: Connectar les dues comunitats!

            # image_token = images[batch][None, :]
            # date_token = dates[batch][None, :]
            #
            # ocr_nodes = torch.cat((ocr_features[visual_idxs], image_token))
            # query_nodes = torch.cat((query_features[text_idxs], date_token))
            ocr_nodes = ocr_features[visual_idxs]
            query_nodes = query_features[text_idxs]

            d_aux = self.hungarian_distance(ocr_nodes, query_nodes)
            d.append(d_aux)

        return torch.stack(d)

