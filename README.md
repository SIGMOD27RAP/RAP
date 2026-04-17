# RAP

This project is a C++ implementation of the paper:
**RaP: A Redirect-and-Patch Framework for Efficient Dynamic Updates in Graph-Based ANN Indexes**.

It provides a dynamic graph-based ANN (Approximate Nearest Neighbor) test program built on top of the RaP/HNSW code in this repository.

## Project Structure

- `RaP/`: Core C++ headers and ANN index implementation (including dynamic update related logic).
- `dynamic_test.cpp`: Main executable entry for loading data, building the index, running dynamic delete/insert rounds, and evaluating recall/latency.
- `Makefile`: Build script for compiling `dynamic_test`.
- `run_datasets.sh`: Batch runner for experiments on one or more datasets.
- `dataset/`: Input datasets (base vectors, query vectors, and ground-truth files), expected by the run script.

## Environment Setup

### Requirements

- Linux environment
- `g++` with C++17 support
- OpenMP support (`-fopenmp`)

### Build

From the project root:

```bash
make clean
make
```

This generates the executable:

- `./dynamic_test`

## How to Run

You can run experiments with the provided script

Edit dataset list and parameters in `run_datasets.sh`, then run:

```bash
bash run_datasets.sh
```

The script will:

- build the project,
- iterate over selected datasets,
- run `dynamic_test`,
- write results to `result/<dataset_name>/...csv`.

## Data Format Notes

- Base/query vector files are expected in `.fbin` format:
  - first `int32`: number of vectors,
  - second `int32`: vector dimension,
  - followed by contiguous `float32` vector values.
- Ground-truth path is provided as a prefix and used by the program for loading/caching per-round ground truth files.