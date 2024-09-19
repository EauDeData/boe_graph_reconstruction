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
from src.models.soft_hungarian import SoftHd, SoftHungarian
import math

class TripletLoss(nn.Module):

    def __init__(self, margin=.5, swap=False, reduction='elementwise_mean', dist=False, negative='textual'):
        super(TripletLoss, self).__init__()

        # TODO: DONT HARDCODE; DO HYPERPARAMETER SEARCH
        self.margin = .2
        self.pt_tripplet_loss = torch.nn.TripletMarginLoss(margin = self.margin)
        self.local_hung_contribution = .25
        self.global_hung_contribution = .25
        self.topic_model_contribution = .5

    def forward(self, local_pos_dist, local_neg_dist,
                      global_pos_dist, global_neg_dist,
                      doc_embs, query_embs):

        # HUNGARIAN LOSS FIRST
        local_loss = self.local_hung_contribution * torch.nn.functional.relu(local_pos_dist - local_neg_dist + self.margin)
        global_loss = self.global_hung_contribution * torch.nn.functional.relu(global_pos_dist - global_neg_dist + self.margin)

        rolled_query = torch.roll(query_embs.clone(),
                                  shifts=random.randint(1, query_embs.shape[0] - 1),
                                  dims=0)

        return torch.mean(local_loss + global_loss) + self.pt_tripplet_loss(doc_embs, query_embs, rolled_query)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape becomes (1, max_len, d_model) for batch-first

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # x is [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1)]  # Add positional encoding up to the sequence length
        return self.dropout(x)

class JointModel(nn.Module):
    # TODO: Processar els dos phocs amb
    # PHOCEncoder
    # And the soft hungarian distance
    def __init__(self, vocab_size, embedding_dim, swap=True, device='cuda'):
        super().__init__()

        self.v_cls = torch.nn.Parameter(torch.rand(embedding_dim))
        self.h_vision = FeatureExtractorCNN(embedding_dim)

        vis_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        self.visual_agg =nn.TransformerEncoder(vis_encoder_layer, num_layers=1)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.text_downscale = torch.nn.Linear(2 * embedding_dim, embedding_dim)
        self.vision_project = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(embedding_dim, embedding_dim))

        text_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        self.text_agg =nn.TransformerEncoder(text_encoder_layer, num_layers=1)

        self.hungarian_distance = SoftHd(embedding_dim)
        self.triplet_distance = TripletLoss()
        self.device = device
        self.pos = PositionalEncoding(embedding_dim)
        self.to(device)

    def common_process(self, batch):
        doc_images = batch['images_document'].to(self.device)
        query_images = batch['images_query'].to(self.device)
        query_tokens = batch['query_text_tokens'].to(self.device)
        # print(f'Query images {query_images.shape}, Query tokens: {query_tokens.shape}')

        doc_image_features = self.h_vision(doc_images)
        query_image_features = self.h_vision(query_images)

        query_tokens_embedding = self.embedding(query_tokens)


        dense_doc_features, mask_t1 = to_dense_batch(doc_image_features, batch = batch['idx_images_document'].to(self.device))
        dense_query_features, mask_t2 = to_dense_batch(query_image_features, batch = batch['idx_images_query'].to(self.device))

        return dense_doc_features, dense_query_features, query_tokens_embedding


    def forward(self, batch):

        dense_doc_features, dense_query_features, query_tokens_embedding = self.common_process(batch)

        # WE HAVE SOME DISTANCES HERE TO RUN THE LOSS
        # TODO: USING THEM
        local_hungarian_distances, global_hungarian_distances = self.hungarian_distance(dense_doc_features,
                                                                                        dense_query_features)
        # MIN BATCH = 2 required
        # TODO: CHECK THE ROLLING WORKS AS EXPECTED
        rolled_query = torch.roll(dense_query_features.clone(),
                                  shifts=random.randint(1, dense_query_features.shape[0] - 1),
                                  dims=0)
        negative_local_hungarian_distances, negative_global_hungarian_distances = self.hungarian_distance(
                                                                                        dense_doc_features,
                                                                                        rolled_query
                                                                                        )
        # print(f"Dense query features", dense_query_features.shape)
        visual_cls = self.v_cls.repeat(dense_query_features.shape[0], 1).unsqueeze(1)
        # print('Visual cls shape', visual_cls.shape)
        dense_query_features_with_cls = torch.cat((visual_cls, dense_query_features), dim=1)
        # print(f"visual cls {visual_cls.shape}, pre visual cls dense query {dense_query_features.shape},"
        #       f"post cls {dense_query_features_with_cls.shape}")

        dense_vision_features_with_cls = torch.cat((visual_cls, dense_doc_features), dim=1)

        word_and_lang_features = torch.cat((dense_query_features_with_cls,
                                            query_tokens_embedding), dim = 2)
        # print('word and lang features', word_and_lang_features.shape)

        aggregated_query_features = self.text_agg(self.pos(self.text_downscale(word_and_lang_features)))[:, 0]
        aggregated_doc_features = self.visual_agg(self.pos(self.vision_project(dense_vision_features_with_cls)))[:, 0]
        # print('Agg query features', aggregated_query_features.shape)


        result =  (local_hungarian_distances, global_hungarian_distances,
                negative_local_hungarian_distances, negative_global_hungarian_distances,
                aggregated_doc_features, aggregated_query_features)

        return self.triplet_distance(*result)

    @torch.no_grad()
    def retrieve(self, batch):
        dense_doc_features, dense_query_features, query_tokens_embedding = self.common_process(batch)

        distance_matrix_hd = torch.zeros([dense_doc_features.shape[0]] * 2, device=self.device)

        for i in range(dense_doc_features.shape[0]):
            for j in range(dense_query_features.shape[0]):
                distance_matrix_hd[i, j] = \
                    (sum(self.hungarian_distance(dense_doc_features[i].unsqueeze(0), dense_query_features[j].unsqueeze(0))) / 2).squeeze()

        #
        # # print(f"Dense query features", dense_query_features.shape)
        # visual_cls = self.v_cls.repeat(dense_query_features.shape[0], 1).unsqueeze(1)
        # # print('Visual cls shape', visual_cls.shape)
        # dense_query_features_with_cls = torch.cat((visual_cls, dense_query_features), dim=1)
        # # print(f"visual cls {visual_cls.shape}, pre visual cls dense query {dense_query_features.shape},"
        # #       f"post cls {dense_query_features_with_cls.shape}")
        #
        # dense_vision_features_with_cls = torch.cat((visual_cls, dense_doc_features), dim=1)
        #
        # word_and_lang_features = torch.cat((dense_query_features_with_cls,
        #                                     query_tokens_embedding), dim = 2)
        # # print('word and lang features', word_and_lang_features.shape)
        #
        # aggregated_query_features = self.text_agg(self.pos(self.text_downscale(word_and_lang_features)))[:, 0]
        # aggregated_doc_features = self.visual_agg(self.pos(self.vision_project(dense_vision_features_with_cls)))[:, 0]

        return distance_matrix_hd












