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
