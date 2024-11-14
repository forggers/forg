import torch
from torch import Tensor, nn

from .feature import Feature
from .file import FileFeatures


class Embedding(nn.Module):
    def __init__(self, *, feature: Feature, D: int, width: int, depth: int):
        """
        Uses a multi-layer perceptron to embed a file feature vector into a D-dimensional space.

        Args:
            feature: The model that produced the file features. Uses this model's device.
            D: The output dimension of the embedding.
            width: The width of each hidden layer.
            depth: The number of hidden layers.
        """

        super().__init__()

        self.feature = feature

        in_size = feature.feature_size
        out_size = D

        layers = [nn.Linear(in_size, width), nn.ReLU()]

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, out_size))

        self.mlp = nn.Sequential(*layers)

        self.to(feature.device)

    def forward(self, files: list[FileFeatures]) -> list[Tensor]:
        features = torch.stack([file.features for file in files])
        return self.mlp(features)
