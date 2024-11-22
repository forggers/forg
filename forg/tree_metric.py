import os
from multiprocessing import Pool

import torch
from torch import Tensor

from .file import FileFeatures


def tree_distance(relative_path1: str, relative_path2: str) -> int:
    """Computes file tree distance between two files."""
    path1 = relative_path1.split(os.path.sep)
    path2 = relative_path2.split(os.path.sep)
    # remove common prefix
    while path1 and path2 and path1[0] == path2[0]:
        path1.pop(0)
        path2.pop(0)
    return len(path1) + len(path2)


def tree_distance_matrix(files: list[FileFeatures]) -> Tensor:
    """Given N files, returns an NxN matrix (stored on the cpu)."""
    n = len(files)
    device = files[0].features.device

    print("Computing tree distance matrix...")

    # parallelize tree distance computation
    args = [
        (file1.relative_path, file2.relative_path) for file1 in files for file2 in files
    ]

    with Pool() as pool:
        distances = pool.starmap(tree_distance, args)

    distances = torch.tensor(distances, dtype=torch.int32, device=device)
    matrix = distances.view(n, n)

    return matrix
