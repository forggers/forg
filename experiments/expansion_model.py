from forg import train

for expansion_model_name in ["meta-llama/Meta-Llama-3-8B", "google/gemma-2-2b"]:
    for run in range(3):  # run multiple times
        train(
            "../data/repos/react",
            run_label=f"expansion_model-{expansion_model_name}-run{run}",
            samples=3000,
            epochs=100_000,
            expansion_model_name=expansion_model_name,
            expansion_batch_size=32,  # bigger model -> smaller batch size
        )
