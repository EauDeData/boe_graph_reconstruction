# from src.data.defaults import CLIP_MODEL_TAG
CLIP_MODEL_TAG = None
from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        print(batch.shape)
        x = self.conv1(batch)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        print(x.shape)
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        print(x.shape)
        x = x + self.positional_embedding.to(x.dtype)
        print(x.shape)
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        input('Enter to continue')
        return torch.mean(x, 0)

class RN50(torch.nn.Module):
    def __init__(self, projection_size = 512, device='cuda'):
        super().__init__()

        resnet50 = models.resnet50(pretrained=True)

        self.output_size = projection_size
        self.feature_size = 2048 # self.rn_50_input_output_lut[input_size]

        self.resnet = nn.Sequential(*list(resnet50.children())[:-1])
        self.projection = nn.Sequential(nn.ReLU(), nn.Linear(self.feature_size, projection_size))

        self.device = device
        self.to(self.device)


    def forward(self, batch):

        features = (torch.nn.functional.adaptive_max_pool2d(self.resnet(batch.to(self.device)),
                                                            (1, 1))
                    .view(batch.shape[0], self.feature_size))

        visual_tokens = self.projection(features)


        return visual_tokens


class FeatureExtractorCNN(nn.Module):
    def __init__(self, out_features):
        super(FeatureExtractorCNN, self).__init__()

        # Define the convolutional layers and pooling
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Calculate the size of the output from the feature extractor
        self._to_linear = None
        self.convs = self._get_conv_output((3, 64, 16))  # Assuming input is (1, 64, 16)


        # Define the fully connected layer
        self.fc = nn.Linear(self.convs, out_features)

    def _get_conv_output(self, shape):
        """Helper function to calculate the output size after convolutions"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output_feat = self.features(dummy_input)
            print(f"Feature map shape: {output_feat.size()}")
            return int(np.prod(output_feat.size()))

    def forward(self, x):
        x = self.features(x)  # Pass through the convolutional layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)  # Fully connected layer to get desired output features
        return x

if __name__ == '__main__':
    model = FeatureExtractorCNN(255)

    print(model(torch.rand(1, 3, 64, 16)).shape)
