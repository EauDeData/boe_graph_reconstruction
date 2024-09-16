import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from einops import repeat, rearrange
from sympy.physics.vector import cross
from src.models.attentions import MultiHeadAttention

class AttnModule(torch.nn.Module):
    def __init__(self, emb_dimension):
        super().__init__()
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(emb_dimension, emb_dimension),
            torch.nn.Dropout(0.1)
        )
        self.to_qkv = torch.nn.Linear(emb_dimension, 3 * emb_dimension)

        self.heads = 8
        self.attention = MultiHeadAttention(emb_dimension, num_heads=8)

    def forward(self, tokens_ocr, tokens_query, cross_attn=False):

        qq, kq, vq = self.to_qkv(tokens_query).chunk(3, dim=-1)
        qo, ko, vo = self.to_qkv(tokens_ocr).chunk(3, dim=-1)

        if not cross_attn:
            output_query, attention_query = self.attention(qq, kq, vq)
            output_ocr, attention_ocr = self.attention(qo, ko, vo)
        else:
            # Cross attend features
            output_ocr, attention_query = self.attention(qo, kq, vq)
            output_query, attention_ocr = self.attention(qq, ko, vo)

        return self.to_out(output_ocr), self.to_out(output_query), attention_ocr, attention_query


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
    def __init__(self, emb_dimension, self_attn_layers = 1, cross_attention_layers = 1, td=0.5, ti=0.5, device = 'cuda'):
        super().__init__()

        # For building cliques we will need attention:
        self.device=device
        self.td = td
        self.ti = ti
        self.to(self.device)
        self.emb_dim = emb_dimension
        self.ret_token_query = torch.nn.Parameter(torch.rand(emb_dimension))
        self.ret_token_ocr = torch.nn.Parameter(torch.rand(emb_dimension))

        # NOW HERE THE ZERO SHOT EVERYTHING (SELF AND CROSS ATTENTION)
        self.self_attn_layers_names = []
        for layer_num in range(self_attn_layers):
            layer_name =  f"self_attn_{layer_num}"
            setattr(self, layer_name, AttnModule(emb_dimension))
            self.self_attn_layers_names.append(layer_name)

        self.cross_attn_layers_names = []
        for layer_num in range(cross_attention_layers):
            layer_name =  f"cross_attn_{layer_num}"
            setattr(self, layer_name, AttnModule(emb_dimension))
            self.cross_attn_layers_names.append(layer_name)

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

        dist = torch.matmul(set1, set2.T)

        return dist.abs()

    def hungarian_distance(self, visual_tokens, word_tokens, debugging_batch_visual = None, debugging_batch_words = None):
        '''

        visual_tokens: can be whether post-message passing o pre-message passing
        word_tokens: PRE CONTEXT, otherwise you need context before tagging
        '''
        # Aqui haurem de fer els 2-cliques i 3-cliques amb el mecanisme d'atenció que definim

        cost_matrix = self._cdist(
                                    torch.nn.functional.normalize(visual_tokens, dim=1),
                                    torch.nn.functional.normalize(word_tokens, dim=1)
                                  )
        row_ind, col_ind = linear_sum_assignment(cost_matrix.clone().detach().cpu().numpy())
        if not debugging_batch_visual is None:
            print([(debugging_batch_visual[i], debugging_batch_words[j]) for i,j in zip(row_ind, col_ind)])
            input('Continue?')
        # print(cost_matrix[row_ind, col_ind].mean())
        return cost_matrix[row_ind, col_ind].mean()

    @staticmethod
    def prune_least_attending_tokens(query_attn, x, prune_rate=0.1):
        # Step 1: Extract attention scores to the CLS token (assumed to be the first token)
        cls_attn = query_attn.sum(0)[0, 1:]  # [B, N-1] CLS attention scores (ignores the CLS token itself)
        # Step 2: Compute how many tokens to prune
        total_tokens = cls_attn.shape[0]
        prune_count = int(prune_rate * total_tokens)  # 10% of tokens

        # Step 3: Sort the attention scores for each batch to get the least attending tokens
        _, prune_indices = torch.topk(cls_attn, prune_count, dim=0, largest=False)  # Bottom 10%

        # Step 4: Prune the least attending tokens from the input (x[:, 1:])
        mask = torch.ones_like(cls_attn, dtype=torch.bool)  # Create a mask of ones (True)
        mask.scatter_(0, prune_indices, False)  # Mark the least attending tokens as False (pruned)
        return mask

    def contextualize_and_prune_tokens(self, ocr_features, query_features):

        cls_ocrs = self.ret_token_ocr[None, None, : ] # (1, 1, dims)
        cls_query = self.ret_token_query[None, None, :]

        ocr_features = torch.concatenate((cls_ocrs, ocr_features), dim = 1) # (1, N+1, dims)
        query_features = torch.concatenate((cls_query, query_features), dim = 1) # (1, N+1, dims)

        for layername in self.self_attn_layers_names:

            # This are self attention layers
            layer = getattr(self, layername)
            ocr_features, query_features, ocr_attn, query_attn = layer(ocr_features, query_features)

        query_mask = self.prune_least_attending_tokens(query_attn, query_features)
        ocr_mask = self.prune_least_attending_tokens(ocr_attn, ocr_features)

        for layername in self.cross_attn_layers_names:

            # This are self attention layers
            layer = getattr(self, layername)
            ocr_features,query_features, ocr_attn, query_attn  = layer(ocr_features, query_features, cross_attn=True)

        # TODO: If we want to copy the paper we should prune on cross attention as well
        return ocr_features, query_features, ocr_mask, query_mask

    def forward(self,  ocr_features, query_features, ocr_idxs, query_idxs, return_embs = False):

        d = []
        d_ret = []

        # data = []
        # idxs = []

        for batch, (visual_idxs, text_idxs) in enumerate(zip(ocr_idxs, query_idxs)):
            # TODO: Connectar les dues comunitats!

            # image_token = images[batch][None, :]
            # date_token = dates[batch][None, :]
            #
            # ocr_nodes = torch.cat((ocr_features[visual_idxs], image_token))
            # query_nodes = torch.cat((query_features[text_idxs], date_token))
            ocr_nodes = ocr_features[visual_idxs][None, :] # (1, SEQ, NDIMS)
            query_nodes = query_features[text_idxs][None, :] # (1, SEQ2, NDIMS)

            ocr_nodes, query_nodes, ocr_mask, query_mask  = self.contextualize_and_prune_tokens(ocr_nodes, query_nodes)

            ret_query = query_nodes[0, 0, :]
            ret_ocr = ocr_nodes[0, 0, :]

            d_ret.append(1 - torch.nn.functional.cosine_similarity(ret_query, ret_ocr, dim=0) )

            # We remove those not informative for performing retrieval
            prunned_ocr_nodes = ocr_nodes[0, 1:][ocr_mask].reshape(-1, ocr_nodes.shape[-1])
            prunned_query_nodes = query_nodes[0, 1:][query_mask].reshape(-1, query_nodes.shape[-1])

            d_aux = self.hungarian_distance(prunned_ocr_nodes, prunned_query_nodes)
            d.append(d_aux)

        #     if return_embs:
        #         ocr_expanded_nodes =(
        #             torch.concatenate((torch.zeros(1, 1, self.emb_dim, device=self.device), ocr_nodes[None, :]), dim = 1)
        #         )
        #         query_expanded_nodes =(
        #             torch.concatenate((torch.zeros(1, 1, self.emb_dim, device=self.device), query_nodes[None, :]), dim = 1)
        #         )
        #
        #         ocr_context = self.ocr_encoder(ocr_expanded_nodes)[0, 0]
        #         query_context = self.ocr_encoder(query_expanded_nodes)[0, 0]
        #
        #         data.extend([ocr_context, query_context])
        #         idxs.extend([batch, batch])
        # if return_embs:
        #     return torch.stack(d), torch.stack(data), torch.tensor(idxs, device=self.device, dtype=torch.int64)


        return torch.stack(d), torch.stack(d_ret)