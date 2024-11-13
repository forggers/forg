import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class RawFile:
    """File read from disk but not yet processed."""

    path: str
    name: str
    extension: str
    content: Optional[str]

    def __init__(self, path: str):
        self.path = path

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
    features: Tensor


class Feature(nn.Module):
    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        device: str = "cpu",
        embed_max_chars: int = 1024,
        embed_batch_size: int = 8,
    ):
        super().__init__()

        self.device = device
        self.embed_max_chars = embed_max_chars
        self.embed_batch_size = embed_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=device)

        hidden_size = self.model.config.hidden_size

        # global learned content embedding for binary files
        self.binary_content_embedding = nn.Parameter(
            torch.randn(hidden_size, device=device)
        )

    def forward(self, files: list[RawFile]) -> list[FileFeatures]:
        names = [file.name for file in files]
        extensions = [file.extension for file in files]
        text_contents = [file.content for file in files if file.content is not None]

        name_embeddings = self.embed_str_batched(names)
        extension_embeddings = self.embed_str_batched(extensions)

        text_content_embeddings = self.embed_str_batched(text_contents)
        content_embeddings: list[Tensor] = []

        for file in files:
            if file.content is None:
                content_embeddings.append(self.binary_content_embedding)
            else:
                content_embeddings.append(text_content_embeddings.pop(0))

        return [
            FileFeatures(
                path=file.path,
                features=torch.cat([a, b, c]),
            )
            for file, a, b, c in zip(
                files, name_embeddings, extension_embeddings, content_embeddings
            )
        ]

    def embed_str_batched(self, strings: list[str]) -> list[Tensor]:
        embeddings: list[Tensor] = []
        for i in tqdm(
            range(0, len(strings), self.embed_batch_size),
            desc="Embedding batches",
        ):
            batch = strings[i : i + self.embed_batch_size]
            batch_embeddings = self.embed_str(batch)
            embeddings.extend(batch_embeddings)
        return embeddings

    @torch.no_grad()
    def embed_str(self, strings: list[str]) -> list[Tensor]:
        """
        Returns the average of the token embeddings.
        Takes in a batch of strings and returns a batch of embeddings.
        """

        strings = [s[: self.embed_max_chars] for s in strings]

        inputs = self.tokenizer(
            strings,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        embeddings: Tensor = self.model(**inputs).last_hidden_state
        embeddings = embeddings.mean(dim=1)
        return [embedding for embedding in embeddings]
