import os

from .file import RawFile


def load_files(repo_dir: str) -> list[RawFile]:
    raw_files: list[RawFile] = []

    for dir, _, files in os.walk(repo_dir):
        for file in files:
            path = os.path.join(dir, file)
            raw_files.append(RawFile(path=path, repo_dir=repo_dir))

    return raw_files
