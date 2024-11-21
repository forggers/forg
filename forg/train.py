import random
from typing import Annotated

import torch
import typer
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from .costs import DistanceMSECost
from .embedding import Embedding
from .feature import FeatureExpansion
from .utils import detect_device, load_files


def train(
    repo_dir: str,
    sample_size: int = 1000,
    train_split: float = 0.8,
    epochs: int = 1000,
    lr: float = 1e-4,
    D: Annotated[int, typer.Option(help="Embedding dimension")] = 8,
    width: Annotated[int, typer.Option(help="Embedding MLP width")] = 512,
    depth: Annotated[int, typer.Option(help="Embedding MLP depth")] = 2,
):
    expansion = FeatureExpansion(device=detect_device())
    embedding = Embedding(expansion=expansion, D=D, width=width, depth=depth)

    random.seed(42)
    raw_files = load_files(repo_dir)
    raw_files = random.sample(raw_files, sample_size)

    files = expansion.expand(raw_files)
    train_files = files[: int(sample_size * train_split)]
    test_files = files[int(sample_size * train_split) :]

    train_cost = DistanceMSECost(train_files)
    test_cost = DistanceMSECost(test_files)

    optimizer = torch.optim.Adam(embedding.parameters(), lr=lr)

    writer = SummaryWriter()

    for epoch in tqdm(range(epochs)):
        train_c = train_cost(embedding(train_files))
        train_c.backward()
        optimizer.step()
        optimizer.zero_grad()
        writer.add_scalar("Cost/train", train_c, epoch)

        with torch.no_grad():
            test_c = test_cost(embedding(test_files))
            writer.add_scalar("Cost/test", test_c, epoch)

    writer.close()
    return files, embedding


if __name__ == "__main__":
    typer.run(train)
