import csv
import os

import torch

from forg import DistanceMSECost, TrainResult, load_files, train

raw_test_files = load_files("../data/repos/react")
csv_file_path = "../dev/historical_transfer.csv"


@torch.no_grad()
def evaluate_transfer(result: TrainResult) -> float:
    test_files = result.embedding.expansion.expand(raw_test_files)
    test_cost = DistanceMSECost(result.embedding, result.embedding_metric, test_files)
    return test_cost(test_files).item()


os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

with open(csv_file_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["base_repo", "test_cost"])

    for base_repo in ["react-2016", "react-2018", "react-2020", "react-2022"]:
        print("Training on", base_repo)
        result = train(
            # "../data/repos/react",
            os.path.join("../data/repos/historical", base_repo),
            run_label=f"historical_transfer-{base_repo}",
            samples=3000,
            epochs=100_000,
            expansion_batch_size=64,
        )

        print("Evaluating on current react")
        test_cost = evaluate_transfer(result)

        csv_writer.writerow([base_repo, test_cost])
