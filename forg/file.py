import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RawFile:
    """File read from disk but not yet processed."""

    path: str
    cache_path: str
    name: str
    extension: str
    content: Optional[str]

    def __init__(self, *, path: str, cache_path: str):
        self.path = path
        self.cache_path = cache_path

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
