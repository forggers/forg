import os

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class Feature(nn.Module):
    def __init__(self, model_name: str = "google/gemma-2-2b", device: str = "cpu"):
        super().__init__()

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=device)

    def forward(self, file_path: str):
        file_name = os.path.basename(file_path)
        name, extension = os.path.splitext(file_name)

        with open(file_path, "r") as file:
            content = file.read()

        name_embedding = self.embed_str(name)
        extension_embedding = self.embed_str(extension)
        content_embedding = self.embed_str(content)

        return torch.cat(
            [
                name_embedding,
                extension_embedding,
                content_embedding,
            ]
        )

    def embed_str(self, s: str) -> Tensor:
        """Returns the average of the token embeddings."""
        inputs = self.tokenizer(s, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state[0]
        return embeddings.mean(dim=0)
