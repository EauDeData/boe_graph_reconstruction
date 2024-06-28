from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch.nn as nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle
from pytorch_metric_learning import miners, losses

import torch
from torch_geometric.utils import to_dense_batch
class JointModel(nn.Module):
    def __init__(self, visual, textual, graph, queries_encoder):
        super().__init__()
        self.visual_model = nn.DataParallel(visual)
        self.textual_model = nn.DataParallel(textual)
        self.graph_model = graph
        self.device = visual.device
        self.q_encoder = nn.DataParallel(queries_encoder)
        print("BE CAREFUL; WE ARE USING THE SELECTED NODE; IT IS HARD CODED IN THE DATASET")
    def forward(self, batch):

        text_features = self.textual_model(batch['textual_content'].to(self.device))
        image_features = self.visual_model(batch['images'].to(self.device))

        text_query_features = self.q_encoder(batch['ocr_query'].to(self.device)).cpu()
        visual_query_features = self.visual_model(batch['visual_queries'].to(self.device)).cpu()

        node_features = self.graph_model( image_features,
                                    text_features,
                                    batch['input_indices'],
                                    batch['edges'] )

        unbatched_features = node_features['image_features']

        queries_ocr = self.graph_model.upsale_text(text_query_features.to(self.graph_model.device))
        visual_queries = self.graph_model.upscale_img(visual_query_features.to(self.graph_model.device))
        queries = self.graph_model.activation(torch.cat((visual_queries, queries_ocr), 1))

        return {'document': unbatched_features, 'queries': queries}

class TripletLossWithMining(nn.Module):
    def __init__(self):
        super().__init__()
        self.miner = miners.TripletMarginMiner()
        self.loss_func = losses.TripletMarginLoss()

    def forward(self, h, gt, batch=None):

        proto_labels = torch.arange(gt.shape[0])
        if batch is None:
            proto_labels_emb = proto_labels
        else:
            proto_labels_emb = torch.tensor(batch['batch_indices'])
        proto_labels = torch.cat((proto_labels_emb, proto_labels), dim = 0)
        input_embeddings = torch.cat((h, gt), dim = 0)
        miner_out = self.miner(input_embeddings, proto_labels)
        return self.loss_func(input_embeddings, proto_labels, miner_out)

class SimCLRWithMining(nn.Module):
    def __init__(self):
        super().__init__()
        self.miner = miners.BatchHardMiner()
        self.loss_func = losses.NTXentLoss()

    def forward(self, h, gt):

        proto_labels = torch.arange(h.shape[0])
        proto_labels = torch.cat((proto_labels, proto_labels), dim = 0)
        input_embeddings = torch.cat((h, gt), dim = 0)
        miner_out = self.miner(input_embeddings, proto_labels)
        return self.loss_func(input_embeddings, proto_labels, miner_out)