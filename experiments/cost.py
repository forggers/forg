from forg.train import CostType, EmbeddingMetricType, train

for metric in [EmbeddingMetricType.EUCLIDEAN, EmbeddingMetricType.HYPERBOLIC]:
    for cost in [CostType.DISTANCE_MSE, CostType.TSNE]:
        result = train(
            "../data/repos/react",
            run_label=f"cost-{metric}-{cost}",
            samples=3000,
            epochs=100_000,
            expansion_batch_size=64,
            metric=metric,
            cost=cost,
        )
