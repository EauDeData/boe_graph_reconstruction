from src.data.defaults import CLIP_MODEL_TAG

import torch.nn as nn
import torch
import open_clip

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class CLIPVisionEncoder(nn.Module):
    def __init__(self, clip_model_tag = CLIP_MODEL_TAG, device='cuda'):
        super(CLIPVisionEncoder, self).__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_tag, pretrained='laion2b_s34b_b79k')

        self.conv1 = model.visual.conv1
        self.patch_dropout = model.visual.patch_dropout
        self.ln_pre = model.visual.ln_pre
        self.transformer = model.visual.transformer
        self.class_embedding = model.visual.class_embedding
        self.positional_embedding = model.visual.positional_embedding
        self.device=device

    def forward(self, batch):

        x = self.conv1(batch)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)

        return torch.mean(x, 0)