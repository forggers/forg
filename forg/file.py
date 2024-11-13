import os
from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class RawFile:
    """File read from disk but not yet processed."""

    path: str
    """Path of file on the current system."""
    relative_path: str
    """Path of file relative to the repo directory."""
    name: str
    extension: str
    content: Optional[str]

    def __init__(self, *, path: str, repo_dir: str):
        repo_parent = os.path.dirname(repo_dir)

        self.path = os.path.abspath(path)
        self.relative_path = os.path.relpath(path, repo_parent)

        file_name = os.path.basename(path)
        self.name, self.extension = os.path.splitext(file_name)

        if self.is_binary(path):
            self.content = None
        else:
            with open(path, "r") as file:
                self.content = file.read()

    @staticmethod
    def is_binary(path: str) -> bool:
        with open(path, "rb") as file:
            for byte in file.read(1024):
                if byte == 0:
                    return True
        return False


@dataclass
class FileFeatures:
    path: str
    """Path of file on the current system."""
    relative_path: str
    """Path of file relative to the repo directory."""
    features: Tensor
