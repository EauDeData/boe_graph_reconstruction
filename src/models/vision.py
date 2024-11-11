# from src.data.defaults import CLIP_MODEL_TAG
from src.data.defaults import IMAGE_SIZE

CLIP_MODEL_TAG = None
from torchvision import models
import torch.nn as nn
import torch
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



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
        self.convs = self._get_conv_output([3] + list(IMAGE_SIZE))  # Assuming input is (1, 64, 16)


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

class RN18FeatureExtractor(FeatureExtractorCNN):
    def __init__(self, out_features):
        super(FeatureExtractorCNN, self).__init__()
        print("Using resnet!")

        resnet18 = models.resnet18(pretrained=True)
        # Remove the last fully connected layer by taking all layers except the last
        # This will leave you with a feature extractor that outputs a 512-dimensional feature vector
        self.features = nn.Sequential(*list(resnet18.children())[:-1])

        self._to_linear = None
        self.convs = self._get_conv_output([3] + list(IMAGE_SIZE))  # Assuming input is (1, 64, 16)


        # Define the fully connected layer
        self.fc = nn.Linear(self.convs, out_features)



if __name__ == '__main__':
    RN18FeatureExtractor(10)

    model = FeatureExtractorCNN(255)

    print(model(torch.rand(1, 3, 64, 16)).shape)
