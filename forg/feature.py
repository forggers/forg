from typing import cast

import torch
from dotenv import load_dotenv
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .embedding_cache import EmbeddingCache
from .file import FileFeatures, RawFile


class ModelFactory:
    """
    Loads the model + tokenizer only if necessary.
    The model + tokenizer do not persist outside this instance.
    """

    def __init__(self, *, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def get_model(self):
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_name, device_map=self.device
            )
        return self.model


class FeatureExpansion(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = "google/gemma-2-2b",
        device: torch.device,
        embed_max_chars: int = 1024,
        embed_batch_size: int = 8,
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.embed_max_chars = embed_max_chars
        self.embed_batch_size = embed_batch_size

        load_dotenv()
        config = AutoConfig.from_pretrained(model_name)

        self.feature_size = config.hidden_size * 3

        # global learned content embedding for binary files
        self.binary_content_embedding = nn.Parameter(torch.randn(config.hidden_size))

        self.cache = EmbeddingCache(model_name=model_name)

        self.to(device)

    def expand(self, files: list[RawFile]) -> list[FileFeatures]:
        factory = ModelFactory(model_name=self.model_name, device=self.device)

        names = [file.name for file in files]
        extensions = [file.extension for file in files]

        name_embeddings = self.__embed_str_batched(factory, files, names, "name")
        extension_embeddings = self.__embed_str_batched(
            factory, files, extensions, "extension"
        )

        text_files = [file for file in files if file.content is not None]
        text_contents = [cast(str, file.content) for file in text_files]

        text_content_embeddings = self.__embed_str_batched(
            factory, text_files, text_contents, "content"
        )
        content_embeddings: list[Tensor] = []

        for file in files:
            if file.content is None:
                # placeholder zeros for binary files
                content_embeddings.append(
                    torch.zeros_like(self.binary_content_embedding)
                )
            else:
                content_embeddings.append(text_content_embeddings.pop(0))

        return [
            FileFeatures(
                path=file.path,
                relative_path=file.relative_path,
                is_binary=file.content is None,
                features=torch.cat([a, b, c]),
            )
            for file, a, b, c in zip(
                files, name_embeddings, extension_embeddings, content_embeddings
            )
        ]

    def to_parameterized_features(self, files: list[FileFeatures]) -> Tensor:
        """
        Returns a shape (n_files, feature_size) tensor where content dimensions of
        binary files are replaced with the global content embedding.
        """
        features: list[Tensor] = []
        for file in files:
            if file.is_binary:
                non_content = file.features[: -self.binary_content_embedding.shape[0]]
                features.append(torch.cat([non_content, self.binary_content_embedding]))
            else:
                features.append(file.features)
        return torch.stack(features)

    def __embed_str_batched(
        self,
        factory: ModelFactory,
        files: list[RawFile],
        strings: list[str],
        embedding_label: str,
    ) -> list[Tensor]:
        cached_embeddings = [
            self.cache.get(
                file,
                embedding_label=embedding_label,
                embedding_input=s,
                device=self.device,
            )
            for file, s in zip(files, strings)
        ]

        uncached_files = [f for f, e in zip(files, cached_embeddings) if e is None]
        uncached_strings = [s for s, e in zip(strings, cached_embeddings) if e is None]
        uncached_embeddings: list[Tensor] = []

        for i in tqdm(
            range(0, len(uncached_strings), self.embed_batch_size),
            desc="Embedding batches",
        ):
            batch_files = uncached_files[i : i + self.embed_batch_size]
            batch_strings = uncached_strings[i : i + self.embed_batch_size]
            batch_embeddings = self.__embed_str(factory, batch_strings)

            uncached_embeddings.extend(batch_embeddings)

            for f, s, emb in zip(batch_files, batch_strings, batch_embeddings):
                self.cache.save(
                    emb, f, embedding_label=embedding_label, embedding_input=s
                )

        embeddings: list[Tensor] = []

        for cached_emb in cached_embeddings:
            if cached_emb is not None:
                embeddings.append(cached_emb)
            else:
                embeddings.append(uncached_embeddings.pop(0))

        return embeddings

    @torch.no_grad()
    def __embed_str(self, factory: ModelFactory, strings: list[str]) -> list[Tensor]:
        """
        Returns the average of the token embeddings.
        Takes in a batch of strings and returns a batch of embeddings.
        """

        strings = [s[: self.embed_max_chars] for s in strings]

        tokenizer = factory.get_tokenizer()
        model = factory.get_model()

        inputs = tokenizer(strings, return_tensors="pt", padding=True).to(self.device)

        embeddings: Tensor = model(**inputs).last_hidden_state
        embeddings = embeddings.mean(dim=1)
        return [embedding for embedding in embeddings]
