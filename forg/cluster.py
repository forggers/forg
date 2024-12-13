import hashlib
import os
import shutil
from enum import Enum
from typing import Annotated, cast

import torch
import typer
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import ClusterNode

from .embedding import Embedding
from .embedding_metric import EmbeddingMetric
from .file import FileFeatures
from .train import load_checkpoint
from .utils import load_files

__all__ = ["Linkage", "DEFAULT_LINKAGE", "cluster_to_disk", "cluster"]


class Linkage(str, Enum):
    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    CENTROID = "centroid"
    MEDIAN = "median"
    WARD = "ward"


DEFAULT_LINKAGE = Linkage.WARD


@torch.no_grad()
def cluster_to_disk(
    embedding: Embedding,
    embedding_metric: EmbeddingMetric,
    files: list[FileFeatures],
    destination_dir: str,
    *,
    num_dirs: int = 10,
    linkage: Linkage = DEFAULT_LINKAGE,
):
    if os.path.exists(destination_dir):
        # safe to delete destination_dir only if it contains only symlinks
        for dirpath, _dirnames, filenames in os.walk(destination_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if not os.path.islink(filepath):
                    raise ValueError(
                        f"Destination directory {destination_dir} is not empty"
                    )
        shutil.rmtree(destination_dir)

    embeddings = embedding(files)
    dist_matrix = embedding_metric.distance_matrix(embeddings)
    dist_matrix = dist_matrix.cpu().detach()

    Z = hierarchy.linkage(dist_matrix, method=linkage)
    root = hierarchy.to_tree(Z)
    root = cast(ClusterNode, root)

    edge_lengths: list[float] = []

    def compute_edge_lengths(node: ClusterNode):
        if node.left is not None:
            edge_lengths.append(node.dist - node.left.dist)
            compute_edge_lengths(node.left)
        if node.right is not None:
            edge_lengths.append(node.dist - node.right.dist)
            compute_edge_lengths(node.right)

    compute_edge_lengths(root)
    edge_lengths.sort(reverse=True)  # longest edge first

    # cut if edge length >= threshold
    cut_edge_threshold = edge_lengths[num_dirs - 1]

    def new_dir_name():
        # short-sha
        return hashlib.sha256(os.urandom(32)).hexdigest()[:8]

    def symlink_file_to_dir(file_idx: int, dir: str):
        file = files[file_idx]
        os.makedirs(dir, exist_ok=True)

        name, extension = os.path.splitext(os.path.basename(file.path))
        cur_name = name
        suffix = 0

        while os.path.exists(os.path.join(dir, f"{cur_name}{extension}")):
            suffix += 1
            cur_name = f"{name}_{suffix}"

        os.symlink(file.path, os.path.join(dir, f"{cur_name}{extension}"))

    def traverse(node: ClusterNode, cur_dir: str):
        if node.is_leaf():
            symlink_file_to_dir(node.id, cur_dir)
            return

        def traverse_child(child: ClusterNode):
            if node.dist - child.dist >= cut_edge_threshold:
                traverse(child, os.path.join(cur_dir, new_dir_name()))
            else:
                traverse(child, cur_dir)

        if node.left:
            traverse_child(node.left)
        if node.right:
            traverse_child(node.right)

    traverse(root, destination_dir)


@torch.no_grad()
def cluster(
    source_dir: str,
    destination_dir: str,
    checkpoint_dir: Annotated[str, typer.Option()],
    num_dirs: int = 10,
    expansion_batch_size: Annotated[
        int, typer.Option(help="Batch size for feature expansion")
    ] = 8,
    linkage: Linkage = DEFAULT_LINKAGE,
):
    checkpoint = load_checkpoint(checkpoint_dir)

    checkpoint.embedding.expansion.embed_batch_size = expansion_batch_size

    cache = checkpoint.embedding.expansion.cache
    cache.cache_dir = os.path.join(cache.cache_dir, "__cluster__")

    raw_files = load_files(source_dir)
    files = checkpoint.embedding.expansion.expand(raw_files)

    cluster_to_disk(
        checkpoint.embedding,
        checkpoint.embedding_metric,
        files,
        destination_dir,
        num_dirs=num_dirs,
        linkage=linkage,
    )


if __name__ == "__main__":
    typer.run(cluster)
