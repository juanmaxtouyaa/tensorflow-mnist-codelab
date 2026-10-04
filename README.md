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

## Running the notebooks

Google Colab is the simplest environment for this project:

1. Open a notebook in Colab.
2. Optionally select a GPU runtime.
3. Run the cells from top to bottom.
4. Treat each notebook as an independent checkpoint in the codelab progression.

The notebooks load MNIST from the public `gs://mnist-public` dataset paths used by the codelab.

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
