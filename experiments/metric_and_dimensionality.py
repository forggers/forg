from forg.train import EmbeddingMetricType, train

for metric in [EmbeddingMetricType.EUCLIDEAN, EmbeddingMetricType.HYPERBOLIC]:
    for D in [2, 4, 6, 8]:
        train(
            "../data/repos/react",
            run_label=f"metric_and_dimensionality-{metric}-D{D}",
            samples=3000,
            epochs=100_000,
            expansion_batch_size=64,
            D=D,
            metric=metric,
        )
