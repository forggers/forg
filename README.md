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
