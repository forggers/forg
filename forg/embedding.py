from torch import Tensor, nn

from .feature import FeatureExpansion
from .file import FileFeatures


class Embedding(nn.Module):
    def __init__(self, *, expansion: FeatureExpansion, D: int, width: int, depth: int):
        """
        Uses a multi-layer perceptron to embed a file feature vector into a D-dimensional space.

        Args:
            feature: The model that produced the file features. Uses this model's device.
            D: The output dimension of the embedding.
            width: The width of each hidden layer.
            depth: The number of hidden layers.
        """

        super().__init__()

        self.expansion = expansion

        in_size = expansion.feature_size
        out_size = D

        layers = [nn.Linear(in_size, width), nn.ReLU()]

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, out_size))

        self.mlp = nn.Sequential(*layers)

        self.to(expansion.device)

    def forward(self, files: list[FileFeatures]) -> Tensor:
        features = self.expansion.to_parameterized_features(files)
        return self.mlp(features)
