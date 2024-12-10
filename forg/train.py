import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

import matplotlib.pyplot as plt
import torch
import typer
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from .costs import DistanceMSECost, TSNECost
from .embedding import Embedding
from .embedding_metric import EmbeddingMetric, EuclideanMetric, HyperbolicMetric
from .feature import ExpansionMode, FeatureExpansion
from .file import FileFeatures
from .utils import detect_device, load_files, save_plt_to_img


@dataclass
class TrainResult:
    embedding: Embedding
    embedding_metric: EmbeddingMetric
    files: list[FileFeatures]
    train_files: list[FileFeatures]
    test_files: list[FileFeatures]


class EmbeddingMetricType(str, Enum):
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"


class CostType(str, Enum):
    DISTANCE_MSE = "distance_mse"
    TSNE = "tsne"


def train(
    repo_dir: str,
    samples: int = 1000,
    train_split: float = 0.8,
    epochs: int = 1000,
    lr: float = 1e-4,
    expansion_model_name: Annotated[
        str, typer.Option(help="Model name for feature expansion")
    ] = "google/gemma-2-2b",
    expansion_batch_size: Annotated[
        int, typer.Option(help="Batch size for feature expansion")
    ] = 8,
    content_expansion_mode: Annotated[
        ExpansionMode,
        typer.Option(help="How to interpret model output for feature expansion"),
    ] = ExpansionMode.HIDDEN_AVG,
    content_expansion_suffix: Annotated[
        str,
        typer.Option(help="Useful for PromptEOL. See FeatureExpansion in feature.py"),
    ] = "",
    D: Annotated[int, typer.Option(help="Embedding dimension")] = 2,
    width: Annotated[int, typer.Option(help="Embedding MLP width")] = 512,
    depth: Annotated[int, typer.Option(help="Embedding MLP depth")] = 2,
    metric: EmbeddingMetricType = EmbeddingMetricType.EUCLIDEAN,
    cost: CostType = CostType.DISTANCE_MSE,
    plot_interval: int = 100,
):
    expansion = FeatureExpansion(
        model_name=expansion_model_name,
        content_expansion_mode=content_expansion_mode,
        content_expansion_suffix=content_expansion_suffix,
        embed_batch_size=expansion_batch_size,
        device=detect_device(),
    )
    embedding = Embedding(expansion=expansion, D=D, width=width, depth=depth)

    if metric == EmbeddingMetricType.EUCLIDEAN:
        embedding_metric = EuclideanMetric()
    elif metric == EmbeddingMetricType.HYPERBOLIC:
        embedding_metric = HyperbolicMetric()

    raw_files = load_files(repo_dir)

    random.seed(42)
    raw_files = random.sample(raw_files, min(samples, len(raw_files)))

    print("# of files:", len(raw_files))

    files = expansion.expand(raw_files)
    train_files = files[: int(len(files) * train_split)]
    test_files = files[int(len(files) * train_split) :]

    if cost == CostType.DISTANCE_MSE:
        CostClass = DistanceMSECost
    elif cost == CostType.TSNE:
        CostClass = TSNECost

    train_cost = CostClass(embedding, embedding_metric, train_files)
    test_cost = CostClass(embedding, embedding_metric, test_files)

    optimizer = torch.optim.Adam(train_cost.parameters(), lr=lr)

    best_test_c = math.inf
    best_epoch = 0
    best_embedding_state = embedding.state_dict()
    best_embedding_metric_state = embedding_metric.state_dict()

    writer = SummaryWriter()

    @torch.no_grad()
    def get_embeddings_img(files: list[FileFeatures]):
        embeddings = embedding(files).cpu().detach()
        plt.figure()
        plt.scatter(embeddings[:, 0], embeddings[:, 1], s=1)
        plt.title("Embeddings")
        plt.xlabel("Dimension 0")
        plt.ylabel("Dimension 1")
        plt.gca().set_aspect("equal")
        return save_plt_to_img()

    for epoch in tqdm(range(epochs)):
        train_c = train_cost(train_files)
        train_c.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("Cost/train", train_c, epoch)

        with torch.no_grad():
            test_c = test_cost(test_files)
            writer.add_scalar("Cost/test", test_c, epoch)

        if isinstance(embedding_metric, HyperbolicMetric):
            writer.add_scalar("Scale", embedding_metric.scale(), epoch)

        if epoch % plot_interval == 0:
            writer.add_image("Embeddings/train", get_embeddings_img(train_files), epoch)
            writer.add_image("Embeddings/test", get_embeddings_img(test_files), epoch)

        if test_c < best_test_c:
            best_test_c = test_c
            best_epoch = epoch
            best_embedding_state = embedding.state_dict()
            best_embedding_metric_state = embedding_metric.state_dict()

    writer.close()

    embedding.load_state_dict(best_embedding_state)
    embedding_metric.load_state_dict(best_embedding_metric_state)

    print("Best epoch:", best_epoch)

    return TrainResult(
        embedding=embedding,
        embedding_metric=embedding_metric,
        files=files,
        train_files=train_files,
        test_files=test_files,
    )


if __name__ == "__main__":
    typer.run(train)
