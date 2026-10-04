# TensorFlow/Keras MNIST Codelab

Completed notebooks for Google's [TensorFlow, Keras and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist) codelab.

The notebooks follow the codelab's progression from a basic one-layer classifier to a convolutional network with dropout and batch normalization.

## Notebooks

| Notebook | Codelab stage | Main model changes |
| --- | --- | --- |
| `keras_01_mnist.ipynb` | Sections 3 and 5 | Baseline softmax classifier trained with SGD |
| `keras_02_mnist_dense.ipynb` | Sections 6 and 7 | Dense ReLU layers and the Adam optimizer |
| `keras_03_mnist_dense_lrdecay_dropout.ipynb` | Sections 8 and 9 | Exponential learning-rate decay and 25% dropout |
| `keras_04_mnist_convolutional.ipynb` | Sections 11 and 12 | Three convolutional layers and 40% dropout |
| `keras_05_mnist_batch_norm.ipynb` | Section 13 | Batch normalization, adjusted decay, and 30% dropout |

Sections 1–2 introduce the environment, sections 4 and 10 explain the underlying concepts, and sections 14–15 cover cloud training and the conclusion.

## Environment and data

The notebooks are Jupyter (`.ipynb`) files. They require Python, TensorFlow,
NumPy, Matplotlib, Pillow, and IPython. [`environment.yml`](environment.yml)
records a Python 3.10 / TensorFlow 2.10.1 environment with CUDA 11.2 and
cuDNN 8.1 for a single NVIDIA GPU. This version choice follows the CUDA 11.2
stack in the [CESGA FT3 A100 guide](https://cesga-docs.gitlab.io/ft3-user-guide/gpu_nodes.html)
and TensorFlow's [tested build configurations](https://www.tensorflow.org/install/source#gpu).
The actual driver and module versions must still be checked on FT3 before a GPU
run; the environment has not been tested on that cluster yet.

Run `python scripts/download_mnist.py` once before opening a notebook. The script
downloads the four original MNIST IDX files from the [CVDF mirror used by
TensorFlow Datasets](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py),
checks SHA-256 digests, and expands them into `data/mnist/`. Data is ignored by
Git. Each notebook reads from `data/mnist/` by default, or from the directory
named in `MNIST_DATA_DIR`. Download on a node with internet access before
starting a compute job.

The notebooks call the 10,000-image MNIST **test** split `validation` because
that is how the original codelab names it. Their reported validation accuracy
is therefore test-set accuracy, not an independent validation estimate.

## Running on CESGA FinisTerrae III

CESGA's [module guide](https://cesga-docs.gitlab.io/ft3-user-guide/env_modules.html)
recommends its `cesga/system` Miniconda module and storing Conda environments in
`$STORE`. From the repository directory on FT3, use the installed Miniconda
module version shown by `module spider`:

```bash
module load cesga/system
module spider miniconda3
module load miniconda3/<version-shown-by-spider>
mkdir -p "$STORE/.conda/pkgs" "$STORE/.cache/pip"
export CONDA_PKGS_DIRS="$STORE/.conda/pkgs"
export PIP_CACHE_DIR="$STORE/.cache/pip"
conda env create --prefix "$STORE/envs/mnist-ft3" --file environment.yml
conda activate "$STORE/envs/mnist-ft3"
python -m ipykernel install --user --name mnist-ft3 --display-name "Python (mnist-ft3)"
python scripts/download_mnist.py --dest "$STORE/datasets/mnist"
```

For a GPU session, follow CESGA's [Jupyter instructions](https://cesga-docs.gitlab.io/ft3-user-guide/remote_desktops.html)
to request a compute node and start JupyterLab there. A login node is not a
training node. Set the data path before launching JupyterLab, select the
`Python (mnist-ft3)` kernel, and run cells from top to bottom:

```bash
compute --gpu
module load cesga/system
module load miniconda3/<version-shown-by-spider>
conda activate "$STORE/envs/mnist-ft3"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MNIST_DATA_DIR="$STORE/datasets/mnist"
nvidia-smi
python -c 'import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices("GPU"))'
start_jupyter-lab
```

`compute --gpu` requests a GPU; check `nvidia-smi` to see which model FT3
allocated. CESGA shows this command under its T4 guidance. For an A100,
use CESGA's [A100 resource request guidance](https://cesga-docs.gitlab.io/ft3-user-guide/gpu_nodes.html)
and check `compute --help` for the site's interactive options. TensorFlow must list a GPU
before training; otherwise the notebook may run on CPU. CESGA's Jupyter URL
requires its VPN outside a remote desktop.

## Expected progression

The codelab gives these approximate validation milestones. Results vary between runs because model initialization, shuffling, and dropout are stochastic. The saved output in notebook 03 reports 97.75% validation accuracy for one run with dropout.

| Stage | Expected validation accuracy |
| --- | ---: |
| Baseline softmax model | About 90% |
| Dense ReLU model | About 97% |
| Learning-rate decay stage, before dropout | Above 98% |
| Convolutional model with dropout | Above 99% |
| Batch-normalized model | Up to about 99.5% |

## Compatibility notes

The original codelab targeted TensorFlow 2.2. The notebooks retain the original structure and learning progression, with only small compatibility updates for current environments:

- `np.float` was replaced with the built-in `float` type.
- The current Matplotlib grid argument is used.
- Adam's `learning_rate` argument replaces the old `lr` alias.

## License

The notebooks retain their original Google Apache 2.0 license notices.
