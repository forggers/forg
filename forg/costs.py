import torch
from torch import Tensor

from .file import FileFeatures
from .metric import tree_distance_matrix


class DistanceMSECost:
    def __init__(self, files: list[FileFeatures]):
        self.tree_dist_matrix = tree_distance_matrix(files)
        """Shape: (# files, # files)"""

    def __call__(self, embeddings: Tensor) -> Tensor:
        """Embeddings: shape (# files, embedding_size)"""
        dist_matrix = torch.cdist(embeddings, embeddings)
        squared_error = (dist_matrix - self.tree_dist_matrix) ** 2
        return squared_error.mean()
