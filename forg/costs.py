import torch
from torch import Tensor

from .file import FileFeatures
from .tree_metric import tree_distance_matrix


class DistanceMSECost:
    def __init__(self, files: list[FileFeatures]):
        self.tree_dist_matrix = tree_distance_matrix(files)
        """Shape: (# files, # files)"""

    def __call__(self, embeddings: Tensor) -> Tensor:
        """Embeddings: shape (# files, embedding_size)"""
        dist_matrix = torch.cdist(embeddings, embeddings)
        squared_error = (dist_matrix - self.tree_dist_matrix) ** 2
        return squared_error.mean()


class TSNECost:
    def __init__(self, files: list[FileFeatures], perplexity: float = 30):
        n = len(files)
        tree_dist_matrix = tree_distance_matrix(files)

        # p_{j|i} = exp(-dist^2 / 2 * sigma_i^2) / normalization
        p_asym = torch.exp(-(tree_dist_matrix**2) / (2 * perplexity**2))
        p_asym = p_asym / p_asym.sum(dim=1, keepdim=True)

        # symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2 * N)
        self.p = (p_asym + p_asym.t()) / (2 * n)

    def __call__(self, embeddings: Tensor) -> Tensor:
        """Embeddings: shape (# files, embedding_size)"""
        dist_matrix = torch.cdist(embeddings, embeddings)

        # q_{ij} = (1 + dist^2)^-1 / normalization
        q = 1 / (1 + dist_matrix**2)
        q = q / q.sum(dim=1, keepdim=True)
        kl_div = self.p * torch.log(self.p / q)

        # debug: check for NaNs
        assert not torch.isnan(kl_div).any()

        return kl_div.sum()
