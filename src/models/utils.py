from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch.nn as nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class JointModel(nn.Module):
    def __init__(self, visual, textual, graph):
        super().__init__()
        self.visual_model = nn.DataParallel(visual)
        self.textual_model = nn.DataParallel(textual)
        self.graph_model = graph
        self.device = visual.device

    def forward(self, batch):
        text_features = self.textual_model(batch['textual_content'].to(self.device))
        image_features = self.visual_model(batch['images'].to(self.device))

        node_features = self.graph_model(image_features,
                                    text_features,
                                    batch['input_indices'],
                                    batch['gt_indices'],
                                    batch['edges'],
                                    batch['ars'].to(self.device))
        return node_features