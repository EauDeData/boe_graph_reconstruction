from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch.nn as nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle
from pytorch_metric_learning import miners, losses
import random
import torch
from torch_geometric.utils import to_dense_batch
import open_clip
from src.models.vision import RN50, _expand_token, FeatureExtractorCNN
from src.models.text import PHOCEncoder, StringEmbedding
from src.models.soft_hungarian import SoftHd


class TripletLoss(nn.Module):

    def __init__(self, margin=.5, swap=False, reduction='elementwise_mean', dist=False, negative='textual'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
        self.dist = dist
        assert negative in ['textual', 'visual', 'both'], f"Negative strategy {negative} is not implemented"
        self.neg_strategy = negative
        if self.dist:
            self.lambda_dist = 0.25

        self.miner_func = miners.TripletMarginMiner()
        self.loss_func = losses.TripletMarginLoss()


    # TODO: Arreglar això, aconseguir la negativa
    def forward(self, ocr_grammars, query_grammars, ocr_batch_idxs, query_batch_idxs, distance):

        roll_right = lambda lst: [lst[-1]] + lst[:-1]

        neg_idx = roll_right(ocr_batch_idxs)
        neg_pretext_batch = ocr_grammars.clone()

        return self._forward(ocr_grammars, query_grammars, neg_pretext_batch,
                             ocr_batch_idxs, query_batch_idxs, neg_idx, distance)

    def _forward(self, ocr_grammars, query_grammars, neg_grammars,
                 ocr_batch_idxs, query_batch_idxs, neg_batch_idxs, distance):

        d_pos, d_ret = distance(ocr_grammars, query_grammars, ocr_batch_idxs, query_batch_idxs)
        d_neg, d_ret_neg = distance(ocr_grammars, neg_grammars, ocr_batch_idxs, neg_batch_idxs)
        # if self.swap:
        #     d_neg_aux, d_ret_neg_aux = distance(query_grammars, neg_grammars, query_batch_idxs, neg_batch_idxs)
        #     d_neg = torch.min(d_neg, d_neg_aux)

        loss = torch.clamp(d_pos - d_neg + self.margin, 0.0)
        triplet_loss = torch.clamp(d_ret - d_ret_neg + self.margin, 0.0)
        # print(d_ret, 'minus', d_ret_neg, 'plus', self.margin, '=\n', triplet_loss)
        # input('c: ')
        # miner_output = self.miner_func(ctx_emeddings, ctx_idxs)
        # triplet_loss = self.loss_func(ctx_emeddings, ctx_idxs, miner_output)

        return {'bipartite_loss': loss.mean(), 'context_loss': triplet_loss.mean()}

class JointModel(nn.Module):
    # TODO: Processar els dos phocs amb
    # PHOCEncoder
    # And the soft hungarian distance
    def __init__(self, vocab_size, embedding_dim, swap=True, device='cuda'):
        super().__init__()

        self.distance = SoftHd(embedding_dim, device=device)
        self.loss_function = TripletLoss(swap=swap)

        # This both models are bullshit and we should reconsider
        self.text_encoder = StringEmbedding(embedding_dim, vocab_size)
        self.vision_model = FeatureExtractorCNN(embedding_dim)



        self.device = device
        self.to(device)

    def forward(self, batch):
        # vision_features = self.vision_encoder(batch['images'].to(self.device))
        # date = self.phoc_encoder(batch['dates'].to(self.device))

        query_features = self.text_encoder(batch['queries'].to(self.device))
        ocr_features = self.vision_model(batch['images'].to(self.device))
        loss = self.loss_function(ocr_features, query_features, batch['images_batch_idxs'],
                           batch['queries_batch_idxs'], self.distance)

        return loss

    @torch.no_grad()
    def retrieve(self, batch):
        # vision_features = self.vision_encoder(batch['images'].to(self.device))
        # date = self.phoc_encoder(batch['dates'].to(self.device))

        query_features = self.text_encoder(batch['queries'].to(self.device))
        ocr_features = self.vision_model(batch['images'].to(self.device))

        scores = torch.zeros((len(batch['queries_batch_idxs']), len(batch['images_batch_idxs'])))
        for batch_i in range(scores.shape[0]):

            ocr_idxs_i = [batch['images_batch_idxs'][batch_i]]
            # vision_features_i = vision_features[batch_i][None, :]

            for batch_j in range(scores.shape[1]):
                query_idxs_j = [batch['queries_batch_idxs'][batch_j]]
                # date_features_j = date[batch_j][None, :]

                # self.distance.hungarian_distance(ocr_features[ocr_idxs_i], query_features[query_idxs_j],
                #                                  debugging_batch_visual=batch['ocr_words'][batch_i],
                #                                  debugging_batch_words=batch['query_str'][batch_j])
                scores[batch_i, batch_j] = sum(
                    *self.distance(ocr_features, query_features, ocr_idxs_i, query_idxs_j
                                  )
                )

        return 1 - (scores / scores.max())










