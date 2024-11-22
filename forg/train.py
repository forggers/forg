import random
from enum import Enum
from typing import Annotated

import torch
import typer
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from .costs import DistanceMSECost
from .embedding import Embedding
from .embedding_metric import EuclideanMetric, HyperbolicMetric
from .feature import FeatureExpansion
from .utils import detect_device, load_files


class EmbeddingMetricType(Enum):
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"


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
    D: Annotated[int, typer.Option(help="Embedding dimension")] = 2,
    width: Annotated[int, typer.Option(help="Embedding MLP width")] = 512,
    depth: Annotated[int, typer.Option(help="Embedding MLP depth")] = 2,
    metric: EmbeddingMetricType = EmbeddingMetricType.EUCLIDEAN,
):
    expansion = FeatureExpansion(
        model_name=expansion_model_name,
        embed_batch_size=expansion_batch_size,
        device=detect_device(),
    )
    embedding = Embedding(expansion=expansion, D=D, width=width, depth=depth)

    if metric == EmbeddingMetricType.EUCLIDEAN:
        embedding_metric = EuclideanMetric()
    elif metric == EmbeddingMetricType.HYPERBOLIC:
        embedding_metric = HyperbolicMetric()

    raw_files = load_files(repo_dir)

    if samples < len(raw_files):
        random.seed(42)
        raw_files = random.sample(raw_files, samples)

    print("# of files:", len(raw_files))

    files = expansion.expand(raw_files)
    train_files = files[: int(len(files) * train_split)]
    test_files = files[int(len(files) * train_split) :]

    train_cost = DistanceMSECost(embedding, embedding_metric, train_files)
    test_cost = DistanceMSECost(embedding, embedding_metric, test_files)

    optimizer = torch.optim.Adam(train_cost.parameters(), lr=lr)

    writer = SummaryWriter()

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

    writer.close()
    return files, embedding


if __name__ == "__main__":
    typer.run(train)
