import hashlib
import os
import shutil
from typing import cast

from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree

from .embedding import Embedding
from .embedding_metric import EmbeddingMetric
from .file import FileFeatures


def cluster_to_disk(
    embedding: Embedding,
    embedding_metric: EmbeddingMetric,
    files: list[FileFeatures],
    destination_dir: str,
    num_dirs: int = 10,
):
    shutil.rmtree(destination_dir, ignore_errors=True)

    embeddings = embedding(files)
    dist_matrix = embedding_metric.distance_matrix(embeddings)
    dist_matrix = dist_matrix.cpu().detach()

    Z = linkage(dist_matrix, "ward")
    root = to_tree(Z)
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
