import io
import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor

from .file import RawFile

__all__ = ["detect_device", "empty_cache", "load_files", "save_plt_to_img"]


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


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_files(repo_dir: str) -> list[RawFile]:
    raw_files: list[RawFile] = []

    for dir, _, files in os.walk(repo_dir):
        for file in files:
            path = os.path.join(dir, file)
            raw_files.append(RawFile(path=path, repo_dir=repo_dir))

    return raw_files


def save_plt_to_img() -> Tensor:
    # save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # convert the buffer to an image tensor
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    buf.close()
    return image
