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

    def forward(self, batch):
        text_features = self.textual_model(batch['textual_content'].to(self.device))
        image_features = self.visual_model(batch['images'].to(self.device))

        node_features = self.graph_model( image_features,
                                    text_features,
                                    batch['input_indices'],
                                    batch['edges'] )

        indices = batch['batch_indices']
        unbatched_features = node_features['image_features']

        padded_input, mask = to_dense_batch(unbatched_features, torch.tensor(indices, device=self.device))
        nodes_in_batch = 1 / mask.sum(1)

        padded_input_weighted = (padded_input * nodes_in_batch[:, None, None]).sum(1)
        queries = self.q_encoder(batch['queries'].to(self.device))


        return {'document': padded_input_weighted, 'queries': queries}

class TripletLossWithMining(nn.Module):
    def __init__(self):
        super().__init__()
        self.miner = miners.BatchHardMiner()
        self.loss_func = losses.TripletMarginLoss()

    def forward(self, h, gt):

        proto_labels = torch.arange(h.shape[0])
        proto_labels = torch.cat((proto_labels, proto_labels), dim = 0)
        input_embeddings = torch.cat((h, gt), dim = 0)
        miner_out = self.miner(input_embeddings, proto_labels)
        return self.loss_func(input_embeddings, proto_labels, miner_out)
