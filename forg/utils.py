import os

import torch

from .file import RawFile


def detect_device() -> torch.device:
    """
    Use the GPU if available, otherwise use the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_files(repo_dir: str) -> list[RawFile]:
    raw_files: list[RawFile] = []

    for dir, _, files in os.walk(repo_dir):
        for file in files:
            path = os.path.join(dir, file)
            raw_files.append(RawFile(path=path, repo_dir=repo_dir))

    return raw_files
