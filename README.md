# mldl_htwg

Code accompanying the Machine and Deep Learning Lectures. Last updated: Winter Semester 2025/2026.

# Setup for Deep Learning

This repository uses:

-   the **Data Science Stack** (NumPy, Pandas, Matplotlib, etc.)
-   **Keras 3** with **PyTorch** and/or **JAX** as backend
-   **uv** as the package/environment manager (a much faster, modern
    alternative to plain `pip`) 🚀🚀🚀

To test the 

------------------------------------------------------------------------

## 1. Basic Data Science Stack (generic, all platforms)

### 1.1 Create a virtual environment

We call it `htwg_dl_25` and use `uv` to create it right inside the repository:

``` bash
uv venv htwg_dl_25
```

### 1.2 Activate it

``` bash
source htwg_dl_25/bin/activate
```

### 1.3 Install scientific + Jupyter stack

``` bash
uv pip install numpy scipy pandas matplotlib seaborn tqdm scikit-learn jupyterlab ipykernel
```

------------------------------------------------------------------------

## 2. Deep Learning Stack (generic)

### 2.1 Install Keras 3

``` bash
uv pip install "keras>=3.0"
```

Keras 3 is backend-agnostic and can run on top of **PyTorch** or
**JAX** or TensorFlow (not recommended anymore).
You select the backend in your code **before** importing Keras via:

``` python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or "jax"
import keras
```

------------------------------------------------------------------------

## 3. Platform-specific instructions

The exact installation commands for PyTorch and JAX depend on your
platform, especially if you want GPU/MPS acceleration.

### 3.1 Apple Silicon (M1/M2/M3) -- PyTorch + MPS and JAX + Metal

#### Install PyTorch with MPS (Metal) support

Use the PyTorch CPU wheel index, which includes MPS on Apple Silicon:

``` bash
uv pip install torch torchvision torchaudio   --index-url https://download.pytorch.org/whl/cpu
```

*(Even though the index says `cpu`, these wheels contain the Metal/MPS
backend for Apple Silicon.)*

#### Install JAX with Metal support

On Apple Silicon, use the `jax-metal` plugin:

``` bash
uv pip install jax-metal
```

------------------------------------------------------------------------

### 3.2 Other platforms (Linux, Windows, Intel Mac)

#### PyTorch (CPU-only, generic)

If you are **not** on Apple Silicon, a generic CPU install is:

``` bash
uv pip install torch torchvision torchaudio
```

For CUDA-enabled Linux machines, follow the official instructions on the
PyTorch website.

#### JAX (CPU-only, generic)

For non-Apple-Silicon platforms (or Intel Macs):

``` bash
uv pip install jax jaxlib
```

For CUDA-enabled JAX on Linux, follow the official JAX installation
guide.

---------------------------------------------------------------------
## About **uv**

It's worth considering **uv** as our environment and package manager because it is fast, lightweight, and fully compatible with standard Python workflows. It uses a global package cache, so packages are downloaded and built only once, and subsequent environments simply link to them. This saves both installation time and disk space.


Conceptually, you can think of uv as:

- **a drop-in replacement for pip** (you just prefix commands with `uv`)
- **faster** (Rust-based, with global caching)
- **more reliable** (modern dependency resolver)
- using **standard Python venvs** instead of a separate ecosystem like conda

Example:

- `pip install numpy` → `uv pip install numpy`
- `python -m venv env` → `uv venv env` #create a virtual environment

Everything else works the same — just faster and cleaner.