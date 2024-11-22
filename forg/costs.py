import torch
from torch import Tensor, nn

from .embedding import Embedding
from .embedding_metric import EmbeddingMetric
from .file import FileFeatures
from .tree_metric import tree_distance_matrix


class Cost(nn.Module):
    def __init__(self, embedding: Embedding, embedding_metric: EmbeddingMetric):
        super().__init__()
        self.embedding = embedding
        self.embedding_metric = embedding_metric
        self.to(embedding.expansion.device)

    def cost(self, embeddings: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, files: list[FileFeatures]) -> Tensor:
        embeddings = self.embedding(files)
        return self.cost(embeddings)


class DistanceMSECost(Cost):
    def __init__(
        self,
        embedding: Embedding,
        embedding_metric: EmbeddingMetric,
        files: list[FileFeatures],
    ):
        super().__init__(embedding, embedding_metric)

        self.tree_dist_matrix = tree_distance_matrix(files)
        """Shape: (# files, # files)"""

    def cost(self, embeddings: Tensor) -> Tensor:
        """Embeddings: shape (# files, embedding_size)"""
        dist_matrix = self.embedding_metric.distance_matrix(embeddings)
        squared_error = (dist_matrix - self.tree_dist_matrix) ** 2
        return squared_error.mean()


class TSNECost(Cost):
    def __init__(
        self,
        embedding: Embedding,
        embedding_metric: EmbeddingMetric,
        files: list[FileFeatures],
        perplexity: float = 30,
    ):
        super().__init__(embedding, embedding_metric)

        n = len(files)
        tree_dist_matrix = tree_distance_matrix(files)

        # p_{j|i} = exp(-dist^2 / 2 * sigma_i^2) / normalization
        p_asym = torch.exp(-(tree_dist_matrix**2) / (2 * perplexity**2))
        p_asym = p_asym / p_asym.sum(dim=1, keepdim=True)

        # symmetrize: p_ij = (p_{j|i} + p_{i|j}) / (2 * N)
        self.p = (p_asym + p_asym.t()) / (2 * n)

    def cost(self, embeddings: Tensor) -> Tensor:
        """Embeddings: shape (# files, embedding_size)"""
        dist_matrix = self.embedding_metric.distance_matrix(embeddings)

        # q_{ij} = (1 + dist^2)^-1 / normalization
        q = 1 / (1 + dist_matrix**2)
        q = q / q.sum(dim=1, keepdim=True)
        kl_div = self.p * torch.log(self.p / q)

        # debug: check for NaNs
        assert not torch.isnan(kl_div).any()

        return kl_div.sum()
