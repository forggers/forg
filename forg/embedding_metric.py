# Adapted from https://github.com/axchen7/vcsmc/blob/main/vcsmc/distances.py

import math

import torch
from torch import Tensor, nn


class EmbeddingMetric(nn.Module):
    def normalize(self, embeddings: Tensor) -> Tensor:
        """
        Normalizes a batch of embeddings so they can be displayed in a Euclidean
        representation.
        """
        raise NotImplementedError

    def distance(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        """
        Computes the distance between two batches of non-normalized embeddings.
        """
        raise NotImplementedError

    def distance_matrix(self, embeddings) -> Tensor:
        """
        Computes the (# embeddings x # embeddings) distance matrix
        using the distance function.
        """
        n = embeddings.shape[0]
        # repeat like 123123123...
        flat_embeddings1 = embeddings.repeat(n, 1)
        # repeat like 111222333...
        flat_embeddings2 = embeddings.repeat(1, 1, n).view(n * n, -1)
        flat_distances = self.distance(flat_embeddings1, flat_embeddings2)
        return flat_distances.view(n, n)


class EuclideanMetric(EmbeddingMetric):
    def normalize(self, embeddings: Tensor) -> Tensor:
        return embeddings

    def distance(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        return torch.linalg.norm(embeddings1 - embeddings2, dim=-1)


class HyperbolicMetric(EmbeddingMetric):
    """
    Embeddings are represented in a non-normalized form where 0 < |x| < inf.
    Hence, normalized embeddings can be arbitrarily close to the edge of the
    poincaré disk. The normalization function f(|x|)=sqrt(1-exp(-|x|^2)) makes
    computing distances easier.
    """

    def __init__(
        self,
        *,
        initial_scale: float = 1.0,
        fixed_scale: bool = False,
        scale_floor: float | None = 1e-2,
        epsilon=1e-8,
    ):
        super().__init__()

        self.epsilon = epsilon
        if scale_floor is not None:
            self.scale_floor = torch.Tensor([scale_floor])[0]

        if fixed_scale:
            self.log_scale = math.log(initial_scale)
        else:
            self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))

    def scale(self) -> Tensor | float:
        if isinstance(self.log_scale, Tensor):
            if self.scale_floor is not None:
                return torch.max(self.scale_floor, self.log_scale.exp())
            else:
                return self.log_scale.exp()
        else:
            return math.exp(self.log_scale)

    def normalize(self, embeddings: Tensor) -> Tensor:
        # return a vector with the same direction but with the norm passed
        # through f(|x|)=sqrt(1-exp(-x^2))
        norms = torch.sqrt(torch.sum(embeddings**2, -1) + self.epsilon)
        norms_sq = torch.sum(embeddings**2, -1)
        new_norms = torch.sqrt(1 - torch.exp(-norms_sq) + self.epsilon)

        # avoid division by zero
        unit_embeddings = embeddings / (norms.unsqueeze(-1) + self.epsilon)
        return unit_embeddings * new_norms.unsqueeze(-1)

    def distance(self, embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
        # see https://en.wikipedia.org/wiki/Poincaré_disk_model#Lines_and_distance

        normalized1 = self.normalize(embeddings1)
        normalized2 = self.normalize(embeddings2)

        xy_norm_sq = torch.sum((normalized1 - normalized2) ** 2, -1)

        norms1_sq = torch.sum(embeddings1**2, -1)
        norms2_sq = torch.sum(embeddings2**2, -1)

        # accounting for normalization, get simple formula
        # 1/(1-|x_normalized|^2) = 1/(1-f(|x|)^2) = exp(|x|^2)
        one_minus_x_norm_sq_recip = torch.exp(norms1_sq)
        one_minus_y_norm_sq_recip = torch.exp(norms2_sq)

        delta = 2 * xy_norm_sq * one_minus_x_norm_sq_recip * one_minus_y_norm_sq_recip

        # use larger epsilon to make float32 stable
        distance = torch.acosh(1 + delta + (self.epsilon * 10))

        return distance * self.scale()
