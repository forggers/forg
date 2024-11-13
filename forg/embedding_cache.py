import hashlib
import os

import torch
from torch import Tensor

from .file import RawFile

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "embedding_cache")


class EmbeddingCache:
    def __init__(self, *, model_name: str, cache_dir: str = DEFAULT_CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir

    def get(
        self,
        file: RawFile,
        *,
        embedding_label: str,
        embedding_input: str,
        device: str,
    ) -> Tensor | None:
        cache_file = self.__to_cache_file(file, embedding_label, embedding_input)
        if os.path.exists(cache_file):
            return torch.load(cache_file, weights_only=True, map_location=device)
        else:
            return None

    def save(
        self,
        embedding: Tensor,
        file: RawFile,
        embedding_label: str,
        embedding_input: str,
    ) -> None:
        cache_file = self.__to_cache_file(file, embedding_label, embedding_input)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(embedding.cpu(), cache_file)

    def __to_cache_file(
        self, file: RawFile, embedding_label: str, embedding_input: str
    ) -> str:
        clean_model_name = self.model_name.replace("/", "_")
        input_hash = hashlib.sha256(embedding_input.encode()).hexdigest()
        save_file = f"{embedding_label}_{input_hash}.pt"
        return os.path.join(
            self.cache_dir, clean_model_name, file.cache_path, save_file
        )
