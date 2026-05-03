# Mini Transformer

Implementation of a mini Transformer encoder from scratch in PyTorch for binary sequence classification.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate.fish # remove .fish from the end if you are not using fish shell
pip3 install -r requirements.txt
```

## Run Benchmark

```bash
python3 src/benchmark.py
```

## Project Structure

```
data/          → train, validation, test CSV files
src/           → all Python source files
```