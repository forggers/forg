Forg -- File Organization.

This project uses git submodules for training data. Clone this repository with
the `--recursive`, or run the following command after cloning:

```bash
git submodule update --init --recursive
```

First, create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You will also need to add a Hugging Face token to `.env`. See `.env.example`.
For running on a cloud instance, you may want to set the `HF_HOME` environment
variable to a persistent location (defaults to `~/.cache/huggingface`).

To run the training:

```bash
# list all options
python -m forg.train --help

# example run
python -m forg.train --samples 2000 --epochs 100000 --metric hyperbolic data/repos/react
```

Notes:

- 2000 samples is good for testing; adjust # of epochs as you see fit
- Set `--expansion-batch-size 32` if running on a 40GB GPU for faster feature
  expansion

To track runs, install the Tensorboard VS Code plugin, then use the command
palette to "Launch Tensorboard".
