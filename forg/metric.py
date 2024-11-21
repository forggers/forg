import os

import torch
from torch import Tensor

from .file import FileFeatures


def tree_distance(file1: FileFeatures, file2: FileFeatures) -> int:
    """Computes file tree distance between two files."""
    path1 = file1.relative_path.split(os.path.sep)
    path2 = file2.relative_path.split(os.path.sep)
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

    matrix = torch.zeros(n, n, dtype=torch.int32, device=device)
    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            matrix[i, j] = tree_distance(file1, file2)
    return matrix
