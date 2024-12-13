from forg import ExpansionMode, train

train(
    "../data/repos/react",
    run_label=f"content_expansion-hidden-avg-no-suffix",
    samples=3000,
    epochs=100_000,
    expansion_batch_size=64,
    content_expansion_mode=ExpansionMode.HIDDEN_AVG,
    content_expansion_suffix="",
)


train(
    "../data/repos/react",
    run_label=f"content_expansion-hidden-last-no-suffix",
    samples=3000,
    epochs=100_000,
    expansion_batch_size=64,
    content_expansion_mode=ExpansionMode.HIDDEN_LAST,
    content_expansion_suffix="",
)


train(
    "../data/repos/react",
    run_label=f"content_expansion-hidden-last-with-suffix",
    samples=3000,
    epochs=100_000,
    expansion_batch_size=64,
    content_expansion_mode=ExpansionMode.HIDDEN_LAST,
    content_expansion_suffix="\n\n-----\n\nThe file above most likely belongs to a folder named (one word): ",
)
