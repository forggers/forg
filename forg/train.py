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
    expansion_model_name: Annotated[
        str, typer.Option(help="Model name for feature expansion")
    ] = "google/gemma-2-2b",
    expansion_batch_size: Annotated[
        int, typer.Option(help="Batch size for feature expansion")
    ] = 8,
    D: Annotated[int, typer.Option(help="Embedding dimension")] = 2,
    width: Annotated[int, typer.Option(help="Embedding MLP width")] = 512,
    depth: Annotated[int, typer.Option(help="Embedding MLP depth")] = 2,
):
    expansion = FeatureExpansion(
        model_name=expansion_model_name,
        embed_batch_size=expansion_batch_size,
        device=detect_device(),
    )
    embedding = Embedding(expansion=expansion, D=D, width=width, depth=depth)

    raw_files = load_files(repo_dir)

    if sample_size < len(raw_files):
        random.seed(42)
        raw_files = random.sample(raw_files, sample_size)

    print("# of files:", len(raw_files))

    files = expansion.expand(raw_files)
    train_files = files[: int(len(files) * train_split)]
    test_files = files[int(len(files) * train_split) :]

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
