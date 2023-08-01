# Function-Space Regularization in Neural Networks: A Probabilistic Perspective

This repository will contain the official implementation for

**_Function-Space Regularization in Neural Networks: A Probabilistic Perspective_**; Tim G. J. Rudner, Sanyam Kapoor, Shikai Qiu, Andrew Gordon Wilson. **ICML 2023**.

**Abstract:** Parameter-space regularization in neural network optimization is a fundamental tool for improving generalization. However, standard parameter-space regularization methods make it challenging to encode explicit preferences about desired predictive functions into neural network training. In this work, we approach regularization in neural networks from a probabilistic perspective and show that by viewing parameter-space regularization as specifying an empirical prior distribution over the model parameters, we can derive a probabilistically well-motivated regularization technique that allows explicitly encoding information about desired predictive functions into neural network training. This method---which we refer to as function-space empirical Bayes (FS-EB)---includes both parameter- and function-space regularization, is mathematically simple, easy to implement, and incurs only minimal computational overhead compared to standard regularization techniques. We evaluate the utility of this regularization technique empirically and demonstrate that the proposed method leads to near-perfect semantic shift detection, highly-calibrated predictive uncertainty estimates, successful task adaption from pre-trained models, and improved generalization under covariate shift.

<p align="center">
  &#151; <a href="https://timrudner.com/fseb"><b>View Paper</b></a> &#151;
</p>


## Environment Setup

First, set up the conda environment using the conda environment `.yml` files in the repository root, using

```
conda env create -f environment.yml
```

### Installing JAX

To install `jax` and `jaxlib`, use
```
pip install "jax[cuda11_cudnn86]==0.4.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Installing PyTorch (CPU)

To install `pytorch` and `torchvision`, use

```
pip install torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu
```

NB: We recommend installing PyTorch for CPU-only to make sure that PyTorch does not interfere with JAX's memory allocation.


## Running Experiments

NB: Replace `path/to/repo` below by the absolute path to this repository.

**FMNIST**

To run FSEB with a small CNN on FashionMNIST, execute

```
python trainer_nn.py --config configs/fseb-cnn-fmnist.json --config_id 0 --cwd path/to/repo
```

**CIFAR-10**

To run FSEB with a ResNet-18 on CIFAR-10, execute

```
python trainer_nn.py --config configs/fseb-resnet18-cifar10.json --config_id 0 --cwd path/to/repo
```

**Additional config files will be added shortly.**

NB: The configs above do not use additional datasets to construct the context set. Instead, they use a corrupted (i.e., augmented) training set. To use another dataset (e.g., KMNIST or CIFAR-100), change the `--context_points` arg in the config.


## CIFAR-10 Corrupted Evaluation

By default, no CIFAR-10 Corrupted datasets are loaded. To evaluate model performance on these datasets, add `--full_eval` to the configs above and make the following manual modification to the `timm` library:

To load CIFAR-10 Corrupted configurations from TFDS, the following path is necessary in the `timm` library [parser_factory.py](https://github.com/rwightman/pytorch-image-models/blob/v0.6.7/timm/data/parsers/parser_factory.py#L9):

```diff
- name = name.split('/', 2)
+ name = name.split('/', 1)
```

if you do not make this change and use `full_eval` in the config, you will get an error stating

```
No builder could be found in the directory: ./data/CIFAR10 for the builder: speckle_noise_1.
```


## Out-of-Memory Errors

If you encounter an out-of-memory error, you may have to adjust the amount of pre-allocated memory used by jax. This can be done, for example, by setting

```
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"
```

Note that this is only one of many reasons why you may encounter an OOM error.
