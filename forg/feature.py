import os
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class FileFeatures:
    path: str
    features: Tensor


class Feature(nn.Module):
    def __init__(self, model_name: str = "google/gemma-2-2b", device: str = "cpu"):
        super().__init__()

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=device)

        hidden_size = self.model.config.hidden_size

        # global learned content embedding for binary files
        self.binary_content_embedding = nn.Parameter(
            torch.randn(hidden_size, device=device)
        )

    def forward(self, file_path: str) -> FileFeatures:
        file_name = os.path.basename(file_path)
        name, extension = os.path.splitext(file_name)

        with open(file_path, "r") as file:
            content = file.read()

        name_embedding = self.embed_str(name)
        extension_embedding = self.embed_str(extension)

        if self.is_binary(file_path):
            content_embedding = self.binary_content_embedding
        else:
            content_embedding = self.embed_str(content)

        features = [
            name_embedding,
            extension_embedding,
            content_embedding,
        ]

        return FileFeatures(
            path=file_path,
            features=torch.cat(features),
        )

    def is_binary(self, file_path: str) -> bool:
        with open(file_path, "rb") as file:
            for byte in file.read(1024):
                if byte == 0:
                    return True
        return False

    def embed_str(self, s: str) -> Tensor:
        """Returns the average of the token embeddings."""
        inputs = self.tokenizer(s, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state[0]
        return embeddings.mean(dim=0)
