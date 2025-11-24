# mldl_htwg
Code accompaning the Machine and Deep Learning Lectures


# Setup for Deep Learning

Basically we need the "Data Science Stack" and Keras using PyTorch and/or JAX.

### Install the basic Data Science Stack

1. Create a virtual environment. We call it "htwg_dl_25" and use uv to create it. uv is a python package manager which is similar to pip but is much faster 🚀🚀🚀.

```
uv venv htwg_dl_25 
```

2. Activate it
```
source htwg_dl_25/bin/activate
```

3. Install Basic scientific + Jupyter stack
```
uv pip install \
    numpy scipy pandas matplotlib seaborn tqdm\
    scikit-learn jupyterlab ipykernel
```

### Deep Learning Stack
Install Keras
```
uv pip install "keras>=3.0"
```


#### Install PyTorch
Install pytorch for a MAC with Apple Silicon (MPS) or CPU

```
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu
```

#### Install JAX
```
uv pip install jax-metal
```







