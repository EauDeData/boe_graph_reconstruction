from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch.nn as nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle
from pytorch_metric_learning import miners, losses
import random
import torch
from torch_geometric.utils import to_dense_batch
import open_clip
from src.models.vision import RN50, _expand_token
from src.models.text import PHOCEncoder
from src.models.soft_hungarian import SoftHd


class TripletLoss(nn.Module):

    def __init__(self, margin=1.0, swap=False, reduction='elementwise_mean', dist=False, negative='textual'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap
        self.reduction = reduction
        self.dist = dist
        assert negative in ['textual', 'visual', 'both'], f"Negative strategy {negative} is not implemented"
        self.neg_strategy = negative
        if self.dist:
            self.lambda_dist = 0.25


    # TODO: Arreglar això, aconseguir la negativa
    def forward(self, ocr_grammars, query_grammars, ocr_batch_idxs, query_batch_idxs, dates, images, distance):

        roll_right = lambda lst: [lst[-1]] + lst[:-1]

        if self.neg_strategy == 'both':
            neg_strategy = random.choice(['textual', 'visual'])
        else: neg_strategy = self.neg_strategy
        if neg_strategy == 'textual':

            neg_idx = roll_right(query_batch_idxs)
            neg_pretext_batch = torch.roll(query_grammars, 1, 0)

        else:

            neg_idx = roll_right(ocr_grammars)
            neg_pretext_batch = ocr_grammars.clone()

        neg_date = torch.roll(dates, 1, 0)

        return self._forward(ocr_grammars, query_grammars, neg_pretext_batch,
                             ocr_batch_idxs, query_batch_idxs, neg_idx, images, dates, neg_date, distance)

    def _forward(self, ocr_grammars, query_grammars, neg_grammars,
                 ocr_batch_idxs, query_batch_idxs, neg_batch_idxs, images, dates, neg_dates, distance):

        #  ocr_features, query_features, ocr_idxs, query_idxs, images, dates
        d_pos = distance(ocr_grammars, query_grammars, ocr_batch_idxs, query_batch_idxs, images, dates)
        d_neg = distance(ocr_grammars, neg_grammars, ocr_batch_idxs, neg_batch_idxs, images, neg_dates)
        if self.swap:
            d_neg_aux = distance(query_grammars, neg_grammars, query_batch_idxs, neg_batch_idxs, dates, neg_dates)
            d_neg = torch.min(d_neg, d_neg_aux)

        loss = torch.clamp(d_pos - d_neg + self.margin, 0.0)

        return loss.mean()




class JointModel(nn.Module):
    # TODO: Processar els dos phocs amb
    # PHOCEncoder
    # And the soft hungarian distance
    def __init__(self, phoc_dim, embedding_dim, swap=True, device='cuda'):
        super().__init__()

        self.distance = SoftHd(embedding_dim, device=device)
        self.loss_function = TripletLoss(swap=swap)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')

        self.vision_encoder = model

        self.conv1 = model.visual.conv1
        self.patch_dropout = model.visual.patch_dropout
        self.ln_pre = model.visual.ln_pre
        self.transformer = model.visual.transformer
        self.class_embedding = model.visual.class_embedding
        self.positional_embedding = model.visual.positional_embedding
        self.ln_post = model.visual.ln_post
        self.global_pool = model.visual._global_pool
        self.proj = model.visual.proj

        self.device = device
        self.to(device)

    def forward_visual_open_clip(self, images):

        x = self.conv1(images)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)[:x.shape[1]]

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.transformer(x)

        x = self.ln_post(x)
        pooled, _ = self.global_pool(x)
        pooled = pooled @ self.proj
        return pooled
    def forward(self, batch):
        # vision_features = self.vision_encoder(batch['images'].to(self.device))
        # date = self.phoc_encoder(batch['dates'].to(self.device))

        query_features = self.vision_encoder.encode_text(batch['queries'].to(self.device))
        ocr_features = self.forward_visual_open_clip(batch['images'].to(self.device))

        vision_features, date = torch.zeros((2,3)), torch.zeros((2,3))
        loss = self.loss_function(ocr_features, query_features, batch['images_batch_idxs'],
                           batch['queries_batch_idxs'], date, vision_features, self.distance)

        return loss

    @torch.no_grad()
    def retrieve(self, batch):
        # vision_features = self.vision_encoder(batch['images'].to(self.device))
        # date = self.phoc_encoder(batch['dates'].to(self.device))

        query_features = self.vision_encoder.encode_text(batch['queries'].to(self.device))
        ocr_features = self.forward_visual_open_clip(batch['images'].to(self.device))

        scores = torch.zeros((len(batch['queries_batch_idxs']), len(batch['images_batch_idxs'])))
        for batch_i in range(scores.shape[0]):

            ocr_idxs_i = [batch['images_batch_idxs'][batch_i]]
            # vision_features_i = vision_features[batch_i][None, :]
            vision_features_i = torch.zeros((2,3))
            for batch_j in range(scores.shape[1]):
                query_idxs_j = [batch['queries_batch_idxs'][batch_j]]
                # date_features_j = date[batch_j][None, :]
                date_features_j = torch.zeros((2,3))
                scores[batch_i, batch_j] = self.distance(ocr_features, query_features, ocr_idxs_i, query_idxs_j,
                                                         vision_features_i, date_features_j)

        return 1 - (scores / scores.max())










