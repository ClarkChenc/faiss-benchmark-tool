# Faiss Benchmark

This is a lightweight, modular, and easy-to-use benchmark framework for Faiss.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd faiss-benchmark
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the benchmark, simply execute the `main.py` script:

```bash
python main.py
```

### GPU Support

To run the benchmark with GPU acceleration, use the `--gpu` flag:

```bash
python main.py --gpu
```

## Customization

-   **Change dataset size and dimension:** Modify the `DATASET_SIZE` and `DIMENSION` constants in `main.py`.
-   **Add new index types:** Add the desired Faiss index strings to the `INDEX_TYPES` list in `main.py`. The default list includes:
    - `Flat`: Exact search.
    - `PCAR64,Flat`: PCA reduction to 64 dimensions, then exact search.
    - `IVF1024,Flat`: Inverted file with 1024 centroids, exact search in lists.
    - `IVF1024,PQ8+16`: Inverted file with 1024 centroids, Product Quantization with 8-byte codes and 16-byte coarse quantizer.
    - `HNSW32,Flat`: Hierarchical Navigable Small World graph with 32 neighbors, exact search.
    - `HNSW32,PQ8+16`: HNSW with Product Quantization.