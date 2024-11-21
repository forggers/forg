import random

import torch
from tqdm import tqdm

from .costs import DistanceMSECost
from .embedding import Embedding
from .feature import FeatureExpansion
from .utils import load_files


def train(
    *,
    repo_dir: str,
    sample_size: int = 1000,
    train_split: float = 0.8,
    epochs: int = 1000,
    lr: float = 1e-4,
    device: str = "cpu",
):
    expansion = FeatureExpansion(device=device)
    embedding = Embedding(expansion=expansion, D=8, width=512, depth=2)

    random.seed(42)
    raw_files = load_files(repo_dir)
    raw_files = random.sample(raw_files, sample_size)

    files = expansion.expand(raw_files)
    train_files = files[: int(sample_size * train_split)]
    test_files = files[int(sample_size * train_split) :]

    train_cost = DistanceMSECost(train_files)
    test_cost = DistanceMSECost(test_files)

    optimizer = torch.optim.Adam(embedding.parameters(), lr=lr)

    train_costs: list[float] = []
    test_costs: list[float] = []

    for epoch in tqdm(range(epochs)):
        train_c = train_cost(embedding(train_files))
        train_c.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_costs.append(train_c.item())

        with torch.no_grad():
            test_c = test_cost(embedding(test_files))
            test_costs.append(test_c.item())

    return files, embedding, train_costs, test_costs
