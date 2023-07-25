## Standard libraries
import os
import numpy as np
from PIL import Image
from typing import Any
from collections import defaultdict
import time
import tree
import random as random_py
import functools
from functools import partial
from copy import copy, deepcopy
from typing import (Any, Callable, Iterable, Optional, Tuple, Union, Dict)
import warnings
import h5py
import argparse
from tqdm.auto import tqdm
import json
from pprint import pprint
import re

## Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
# %matplotlib inline
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
rcParams['lines.linewidth'] = 2.0
# Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman'] # need to have latex installed for this to work
rcParams['text.usetex'] = True
plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})
import seaborn as sns
sns.reset_orig()

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
from jax import jit
from jax.config import config
# config.update("jax_debug_nans", True)
# config.update('jax_platform_name', 'cpu')

## Flax
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze
from flax.linen.initializers import lecun_normal

## JAX addons
import optax
import distrax
import neural_tangents as nt
import flaxmodels as fm
from flaxmodels.resnet import ops
from flaxmodels import utils

## Tensorflow
import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
tfd = tfp.distributions

## PyTorch
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, MNIST, KMNIST, ImageNet

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import datasets as sklearn_datasets
import wandb

from pathlib import Path
from timm.data import create_dataset
from torch.utils.data import Dataset, random_split

## Convert from CxHxW to HxWxC for Flax.
chw2hwc_fn = lambda img: img.permute(1, 2, 0)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='fmnist')  # cifar10, cifar10-224, cifar100, fmnist, two-moons
parser.add_argument('--prediction_type', type=str, default='classification')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--context_batch_size", type=int, default=32)
parser.add_argument("--training_dataset_size", type=int, default=0)
parser.add_argument("--context_dataset_size", type=int, default=10000)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--learning_rate_scale_logvar", type=float, default=1)
parser.add_argument("--alpha", type=float, default=1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument('--optimizer_name', type=str, default='sgd')  # sgd, adam, adamw
parser.add_argument('--model_name', type=str, default='ResNet18')  # ResNet9, ResNet18, ResNet18-Pretrained, ResNet50-Pretrained, S4D
parser.add_argument('--method', type=str, default='fsmap')  # fsmap, psmap, fsvi, psvi
parser.add_argument('--reg_type', type=str, default='function_prior')  # function_prior, function_norm, parameter_norm, feature_parameter_norm, function_kl, parameter_kl
parser.add_argument('--forward_points', type=str, default='train')
parser.add_argument('--reg_points', type=str, default='joint')
parser.add_argument('--context_points', type=str, default='train')
parser.add_argument("--context_transform", action="store_true", default=False)
parser.add_argument('--ood_points', type=str, default='')
parser.add_argument("--mc_samples_llk", type=int, default=1)
parser.add_argument("--mc_samples_reg", type=int, default=1)
parser.add_argument("--mc_samples_eval", type=int, default=1)
parser.add_argument("--reg_scale", type=float, default=1)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--prior_mean", type=float, default=0)
parser.add_argument("--prior_var", type=float, default=0)
parser.add_argument("--prior_likelihood_scale", type=float, default=1)
parser.add_argument("--prior_likelihood_f_scale", type=float, default=1)
parser.add_argument("--prior_likelihood_cov_scale", type=float, default=0)
parser.add_argument("--prior_likelihood_cov_diag", type=float, default=0)
parser.add_argument("--likelihood_scale", type=float, default=1)
parser.add_argument("--output_var", action="store_true", default=False)
parser.add_argument("--prior_params_var", type=float, default=1)
parser.add_argument("--init_logvar", type=float, default=-50)
parser.add_argument("--init_final_layer_weights_logvar", type=float, default=-50)
parser.add_argument("--init_final_layer_bias_logvar", type=float, default=-50)
parser.add_argument("--prior_feature_logvar", type=float, default=-50)
parser.add_argument("--prior_precision", type=float, default=0)
parser.add_argument("--pretrained_prior", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--evaluate", action="store_true", default=False)
parser.add_argument("--full_eval", action="store_true", default=False)
parser.add_argument("--restore_checkpoint", action="store_true", default=False)
parser.add_argument('--batch_stats_init_epochs', type=int, default=0)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--debug_print", action="store_true", default=False)
parser.add_argument("--debug_psd", action="store_true", default=False)
parser.add_argument("--log_frequency", type=int, default=2)
parser.add_argument("--save_to_wandb", action="store_true", default=False)
parser.add_argument('--wandb_project', type=str, default='function-space_transfer-learning')
parser.add_argument('--wandb_account', type=str, default='tgjr-research')
parser.add_argument('--gpu_mem_frac', type=float, default=0)
parser.add_argument('--config', type=str, default='')
parser.add_argument('--config_id', type=int, default=0)
parser.add_argument('--config_name', type=str, default='')
parser.add_argument('--cwd', type=str, default='')

args = parser.parse_args()
args_dict = vars(args)

config_file = args.config
config_id = args.config_id
config_name = args.config_name

if config_file != '':
    with open(config_file, 'r') as f:
        config_json = json.load(f)

    configurations = config_json['configurations']
    if config_name == '':
        name = configurations[config_id]['name']
    else:
        name = config_name
    id = configurations[config_id]['id']
    cwd = configurations[config_id]['cwd']
    parser_args_list = configurations[config_id]['args']
    env_args = configurations[config_id]['env']

    def is_float(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
        
    parser_args = {}

    for i in range(len(parser_args_list)):
        if parser_args_list[i].startswith('--'):
            key = parser_args_list[i][2:]
            value = parser_args_list[i+1]
            parser_args[key] = value

    print(f"\nConfig name: {name}")
    print(f"\nConfig id: {id}")
    print(f"\nEnvironment args:\n\n{env_args}")

    for key in parser_args:
        args_dict[key] = parser_args[key]

    for key in parser_args:
        if isinstance(parser_args[key], int):
            args_dict[key] = int(parser_args[key])
        elif isinstance(parser_args[key], str) and parser_args[key].isnumeric():
            args_dict[key] = int(parser_args[key])
        elif isinstance(parser_args[key], str) and is_float(parser_args[key]):
            args_dict[key] = float(parser_args[key])
        elif parser_args[key] == 'True' or parser_args[key] == 'False':
            args_dict[key] = True if parser_args[key] == 'True' else False

    for key in env_args:
        os.environ[key] = env_args[key]

dataset = args_dict["dataset"]
prediction_type = args_dict["prediction_type"]
batch_size = args_dict["batch_size"]
context_batch_size = args_dict["context_batch_size"]
training_dataset_size = args_dict["training_dataset_size"]
context_dataset_size = args_dict["context_dataset_size"]
num_epochs = args_dict["num_epochs"]
learning_rate = args_dict["learning_rate"]
learning_rate_scale_logvar = args_dict["learning_rate_scale_logvar"]
alpha = args_dict["alpha"]
momentum = args_dict["momentum"]
optimizer_name = args_dict["optimizer_name"]
model_name = args_dict["model_name"]
method = args_dict["method"]
reg_type = args_dict["reg_type"]
weight_decay = args_dict["weight_decay"]
context_points = args_dict["context_points"]
forward_points = args_dict["forward_points"]
reg_points = args_dict["reg_points"]
context_transform = args_dict["context_transform"]
ood_points = args_dict["ood_points"]
mc_samples_llk = args_dict["mc_samples_llk"]
mc_samples_reg = args_dict["mc_samples_reg"]
mc_samples_eval = args_dict["mc_samples_eval"]
reg_scale = args_dict["reg_scale"]
prior_mean = args_dict["prior_mean"]
prior_var = args_dict["prior_var"]
prior_likelihood_scale = args_dict["prior_likelihood_scale"]
prior_likelihood_f_scale = args_dict["prior_likelihood_f_scale"]
prior_likelihood_cov_scale = args_dict["prior_likelihood_cov_scale"]
prior_likelihood_cov_diag = args_dict["prior_likelihood_cov_diag"]
likelihood_scale = args_dict["likelihood_scale"]
output_var = args_dict["output_var"]
prior_params_var = args_dict["prior_params_var"]
init_logvar = args_dict["init_logvar"]
init_final_layer_weights_logvar = args_dict["init_final_layer_weights_logvar"]
init_final_layer_bias_logvar = args_dict["init_final_layer_bias_logvar"]
prior_feature_logvar = args_dict["prior_feature_logvar"]
prior_precision = args_dict["prior_precision"]
pretrained_prior = args_dict["pretrained_prior"]
seed = args_dict["seed"]
evaluate = args_dict["evaluate"]
full_eval = args_dict["full_eval"]
restore_checkpoint = args_dict["restore_checkpoint"]
batch_stats_init_epochs = args_dict["batch_stats_init_epochs"]
debug = args_dict["debug"]
debug_print = args_dict["debug_print"]
debug_psd = args_dict["debug_psd"]
log_frequency = args_dict["log_frequency"]
save_to_wandb = args_dict["save_to_wandb"]
wandb_project = args_dict["wandb_project"]
wandb_account = args_dict["wandb_account"]
gpu_mem_frac = args_dict["gpu_mem_frac"]
cwd = args_dict["cwd"]

print(f"\nParser args:\n\n{args_dict}")

print(f"\nCWD: {cwd}")
os.chdir(cwd)

# if gpu_mem_frac != 0:
#     os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(gpu_mem_frac)

# Path to the folder where the datasets are/should be downloaded
DATASET_PATH = "data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "checkpoints"

# Seeding for random operations
print(f"\nSeed: {seed}")
main_rng = random.PRNGKey(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
random_py.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
torch.random.manual_seed(seed)

rng_key = main_rng

if debug:
    config.update('jax_disable_jit', True)

jitter = eps = 1e-6

print(f"\nDevice: {jax.devices()[0]}\n")


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.
  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263
  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins
  Returns:
    ece: Expected Calibration Error
    mce: Maximum Calibration Error
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  return ece, mce


@jax.jit
def accuracy(logits_or_p, Y):
    '''Compute accuracy
    Arguments:
        logits_or_p: (B, d)
        Y: (B,) integer labels.
    '''
    if len(Y) == 0:
        return 0.
    matches = jnp.argmax(logits_or_p, axis=-1) == Y
    return jnp.mean(matches)


@jax.jit
def categorical_nll(logits, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return optax.softmax_cross_entropy_with_integer_labels(logits, Y)


@jax.jit
def categorical_nll_with_softmax(p, Y):
    '''Negative log-likelihood of categorical distribution.
    '''
    return -jnp.sum(jnp.log(p + 1e-10) * jax.nn.one_hot(Y, p.shape[-1]), axis=-1)


@jax.jit
def gaussian_nll(f, Y, likelihood_var):
    '''Negative log-likelihood of Gaussian distribution.
    '''
    likelihood = tfd.Normal(f, likelihood_var ** 0.5)
    nll = jnp.sum(-likelihood.log_prob(Y), -1)
    return nll


@jax.jit
def categorical_entropy(p):
    '''Entropy of categorical distribution.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''
    return -jnp.sum(p * jnp.log(p + eps), axis=-1)


# @jax.jit
def selective_accuracy(p, Y):
    '''Selective Prediction Accuracy
    Uses predictive entropy with T thresholds.
    Arguments:
        p: (B, d)

    Returns:
        (B,)
    '''

    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores_id, threshold)
        mask = np.array(scores_id <= p)
        thresholded_accuracies.append(np.mean(accuracies_test[mask]))
    values_id = np.array(thresholded_accuracies)

    auc_sel_id = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel_id += (x * values_id[i] + x * values_id[i+1]) / 2

    return auc_sel_id


def selective_accuracy_test_ood(p_id, p_ood, Y):
    thresholds = np.concatenate([np.linspace(100, 1, 100), np.array([0.1])], axis=0)

    predictions_test = p_id.argmax(-1)
    accuracies_test = predictions_test == Y
    scores_id = categorical_entropy(p_id)

    accuracies_ood = jnp.zeros(p_ood.shape[0])
    scores_ood = categorical_entropy(p_ood)

    accuracies = jnp.concatenate([accuracies_test, accuracies_ood], axis=0)
    scores = jnp.concatenate([scores_id, scores_ood], axis=0)

    thresholded_accuracies = []
    for threshold in thresholds:
        p = np.percentile(scores, threshold)
        mask = np.array(scores <= p)
        thresholded_accuracies.append(np.mean(accuracies[mask]))
    values = np.array(thresholded_accuracies)

    auc_sel = 0
    for i in range(len(thresholds)-1):
        if i == 0:
            x = 100 - thresholds[i+1]
        else:
            x = thresholds[i] - thresholds[i+1]
        auc_sel += (x * values[i] + x * values[i+1]) / 2

    return auc_sel


def auroc_logits(predicted_logits_test, predicted_logits_ood, score, rng_key):
    predicted_targets_test = jax.nn.softmax(predicted_logits_test, axis=-1)
    predicted_targets_ood = jax.nn.softmax(predicted_logits_ood, axis=-1)

    ood_size = predicted_targets_ood.shape[1]
    test_size = predicted_targets_test.shape[1]
    anomaly_targets = jnp.concatenate((np.zeros(test_size), np.ones(ood_size)))
    if score == "entropy":
        entropy_test = categorical_entropy(predicted_targets_test.mean(0))
        entropy_ood = categorical_entropy(predicted_targets_ood.mean(0))
        scores = jnp.concatenate((entropy_test, entropy_ood))
    if score == "expected entropy":
        entropy_test = categorical_entropy(predicted_targets_test).mean(0)
        entropy_ood = categorical_entropy(predicted_targets_ood).mean(0)
        scores = jnp.concatenate((entropy_test, entropy_ood))
    elif score == "mutual information":
        mutual_information_test = categorical_entropy(predicted_targets_test.mean(0)) - categorical_entropy(predicted_targets_test).mean(0)
        mutual_information_ood = categorical_entropy(predicted_targets_ood.mean(0)) - categorical_entropy(predicted_targets_ood).mean(0)
        scores = jnp.concatenate((mutual_information_test, mutual_information_ood))
    else:
        NotImplementedError
    fpr, tpr, _ = roc_curve(anomaly_targets, scores)
    auroc_score = roc_auc_score(anomaly_targets, scores)
    return auroc_score


def merge_params(params_1, params_2):
    flat_params_1 = flax.traverse_util.flatten_dict(params_1)
    flat_params_2 = flax.traverse_util.flatten_dict(params_2)
    flat_params = flat_params_1 | flat_params_2
    unflat_params = flax.traverse_util.unflatten_dict(flat_params)
    return unflat_params


def split_params(params, type="dense"):
    flat_params_fixed = flax.traverse_util.flatten_dict(params)
    flat_params_rest = flax.traverse_util.flatten_dict(params)
    keys = flat_params_fixed.keys()

    i = -1
    for key in list(keys):
        if "Dense" in str(key) and "kernel" in str(key):
            i += 1

    if type == "dense":
        for key in list(keys):
            if f"Dense_{i}" in str(key):  # first check if there may be two final dense layers
                flat_params_fixed.pop(key)
            else:
                flat_params_rest.pop(key)
    elif type == "batch_norm":
        for key in list(keys):
            if "BatchNorm" in str(key):
                flat_params_fixed.pop(key)
            else:
                flat_params_rest.pop(key)
    else:
        raise NotImplementedError
        
    unflat_params_fixed = flax.traverse_util.unflatten_dict(flat_params_fixed)
    unflat_params_fixed = unflat_params_fixed
    unflat_params_rest = flax.traverse_util.unflatten_dict(flat_params_rest)
    unflat_params_rest = unflat_params_rest

    return unflat_params_fixed, unflat_params_rest


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class CustomDataset(Dataset):
    def __init__(self, original_dataset, desired_size):
        self.original_dataset = original_dataset
        self.desired_size = desired_size

    def __len__(self):
        return self.desired_size

    def __getitem__(self, idx):
        idx = idx % len(self.original_dataset)  # wrap around the original dataset
        return self.original_dataset[idx]


class TwoMoons(torch.utils.data.Dataset):
    def __init__(self, size: int, inputs: np.ndarray, targets: np.ndarray):
        self.size = size
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input, target = self.inputs[index], int(self.targets[index][0].sum())
        ret = (input, target)
        return ret


def get_cifar10_test(root=None, v1=False, corr_config=None, batch_size=128, **_):
    _TEST_TRANSFORM = [
        transforms.ToTensor(),
        transforms.Normalize((.4914, .4822, .4465), (.247, .243, .261)),
    ]

    if dataset == 'cifar10-224':
        _TEST_TRANSFORM.append(transforms.Resize(224))

    _TEST_TRANSFORM.append(transforms.Lambda(chw2hwc_fn))

    if v1:
        test_data = create_dataset(name='tfds/cifar10_1', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)
    elif corr_config is not None:
        test_data = create_dataset(f'tfds/cifar10_corrupted/{corr_config}', root=root, split='test',
                                    is_training=True, batch_size=batch_size,
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)
    else:
        test_data = create_dataset('torch/cifar10', root=root, split='test',
                                    transform=transforms.Compose(_TEST_TRANSFORM), download=True)

    return test_data

num_workers_test = 4
persistent_workers_test = True

if dataset == 'cifar10' or dataset == 'cifar10-224':
    # _train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    _train_dataset = CIFAR10(root="./data/CIFAR10", train=True, download=True)
    input_dim = 32
    num_classes = 10
    dataset_size = 50000
    testset_size = 10000
    if training_dataset_size == 0:
        training_dataset_size = 50000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size
    
    # DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2))
    # DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2))
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.2023, 0.1994, 0.2010])
    DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    DATA_STD = np.array([0.247, 0.243, 0.261])
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    test_transform_list = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]
    train_transform_list = [
        transforms.RandomCrop(input_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]

    if dataset == 'cifar10-224':
        test_transform_list.append(transforms.Resize(224))
        train_transform_list.append(transforms.Resize(224))

    test_transform_list.append(image_to_numpy)
    train_transform_list.append(image_to_numpy)

    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    _train_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=train_transform, download=True)
    # val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    val_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=test_transform, download=True)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = CustomDataset(train_dataset, dataset_size)
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    # test_dataset = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)
    test_dataset = CIFAR10(root="./data/CIFAR10", train=False, transform=test_transform, download=True)

    if False:  # context_points == "imagenet":
        DATA_MEANS_CONTEXT = np.array([0.485, 0.456, 0.406])
        DATA_STD_CONTEXT = np.array([0.229, 0.224, 0.225])
    else:
        DATA_MEANS_CONTEXT = DATA_MEANS
        DATA_STD_CONTEXT = DATA_STD

    def image_to_numpy_context(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS_CONTEXT) / DATA_STD_CONTEXT
        return img

    if context_transform:
        context_transform_list = [
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=(3,3)),
            # transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(32),
        ]
        if dataset == 'cifar10-224':
            if context_points == "imagenet":
                context_transform_list.append(transforms.RandomResizedCrop(224))
            else:
                context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy_context)
        context_transform = transforms.Compose(context_transform_list)
    else:
        if context_points == "imagenet":
            context_transform_list = [
                transforms.RandomResizedCrop(224),
                # transforms.Resize(32),
                # transforms.Resize(224),
            ]
            context_transform_list.append(image_to_numpy_context)
            context_transform = transforms.Compose(context_transform_list)
        else:
            context_transform = test_transform

    if context_points == "train":
        context_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=context_transform, download=True)
    elif context_points == "cifar100":
        context_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=context_transform, download=True)
    elif context_points == "svhn":
        context_dataset = SVHN(root="./data/SVHN", split="train", download=True, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", transform=context_transform)
    else:
        ValueError("Unknown context dataset")
    
    full_context_dataset_size = len(context_dataset)
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, full_context_dataset_size - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)

    if dataset == 'cifar10-224':
        ood_transform = transforms.Compose([
            transforms.Resize(224),
            image_to_numpy,
        ])
    else:
        ood_transform = test_transform

    if ood_points == "svhn":
        ood_dataset = SVHN(root="./data/SVHN", split="test", download=True, transform=ood_transform)
    elif ood_points == "cifar100":
        ood_dataset = CIFAR100(root="./data/CIFAR100", train=False, download=True, transform=ood_transform)
    else:
        ValueError("Unknown OOD dataset")

    ood_dataset = CustomDataset(ood_dataset, len(test_dataset))
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )
    
    cifar101test_data = get_cifar10_test(root="./data/CIFAR10", seed=seed, v1=True, corr_config=None, batch_size=batch_size_test)

    cifar101test_loader  = data.DataLoader(cifar101test_data,
                                batch_size=batch_size_test,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers_test,
                                persistent_workers=persistent_workers_test
                                )

    try:
        if full_eval:
            corr_config_list = [
                "speckle_noise_1", "speckle_noise_2", "speckle_noise_3", "speckle_noise_4", "speckle_noise_5",
                "shot_noise_1", "shot_noise_2", "shot_noise_3", "shot_noise_4", "shot_noise_5",
                "pixelate_1", "pixelate_2", "pixelate_3", "pixelate_4", "pixelate_5",
                "gaussian_blur_1", "gaussian_blur_2", "gaussian_blur_3", "gaussian_blur_4", "gaussian_blur_5",
                ]
            ccifar10test_loader_list = []
            for corr_config in corr_config_list:
                ccifar10test_data = get_cifar10_test(root="./data/CIFAR10", seed=seed, v1=False, corr_config=corr_config, batch_size=batch_size_test)

                ccifar10test_loader  = data.DataLoader(ccifar10test_data,
                                            batch_size=batch_size_test,
                                            shuffle=False,
                                            drop_last=False,
                                            num_workers=num_workers_test,
                                            persistent_workers=persistent_workers_test
                                            )
                ccifar10test_loader_list.append(ccifar10test_loader)
    except:
        print("Could not load corrupted CIFAR10 datasets.")
        assert full_eval == False, "Could not load corrupted CIFAR10 datasets. Please set full_eval to False or modify timm library to enabling loading datasets."

elif dataset == 'cifar100' or dataset == 'cifar100-224':
    _train_dataset = CIFAR100(root="./data/CIFAR100", train=True, download=True)
    input_dim = 32
    num_classes = 100
    dataset_size = 50000
    testset_size = 10000
    if training_dataset_size == 0:
        training_dataset_size = 50000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2))
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2))
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.2023, 0.1994, 0.2010])
    # DATA_MEANS = np.array([0.4914, 0.4822, 0.4465])
    # DATA_STD = np.array([0.247, 0.243, 0.261])
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    test_transform_list = [
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]
    train_transform_list = [
        transforms.RandomCrop(input_dim, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.ToTensor(),
    ]

    if dataset == 'cifar100-224':
        test_transform_list.append(transforms.Resize(224))
        train_transform_list.append(transforms.Resize(224))

    test_transform_list.append(image_to_numpy)
    train_transform_list.append(image_to_numpy)

    test_transform = transforms.Compose(test_transform_list)
    train_transform = transforms.Compose(train_transform_list)

    _train_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=train_transform, download=True)
    # val_dataset = CIFAR100(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    val_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=test_transform, download=True)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = CustomDataset(train_dataset, dataset_size)
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = CIFAR100(root="./data/CIFAR100", train=False, transform=test_transform, download=True)

    if False: #  context_points == "imagenet":
        DATA_MEANS_CONTEXT = np.array([0.485, 0.456, 0.406])
        DATA_STD_CONTEXT = np.array([0.229, 0.224, 0.225])
    else:
        DATA_MEANS_CONTEXT = DATA_MEANS
        DATA_STD_CONTEXT = DATA_STD

    def image_to_numpy_context(img):
        img = np.array(img, dtype=np.float32)
        img = (img / 255. - DATA_MEANS_CONTEXT) / DATA_STD_CONTEXT
        return img
    
    if context_transform:
        context_transform_list = [
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(32),
        ]
        if dataset == 'cifar100-224':
            if context_points == "imagenet":
                context_transform_list.append(transforms.RandomResizedCrop(224))
            else:
                context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy_context)
        context_transform = transforms.Compose(context_transform_list)
    else:
        if context_points == "imagenet":
            context_transform_list = [
                transforms.RandomResizedCrop(224),
                # transforms.Resize(32),
                # transforms.Resize(224),
            ]
            context_transform_list.append(image_to_numpy_context)
            context_transform = transforms.Compose(context_transform_list)
        else:
            context_transform = test_transform

    if context_points == "train":
        context_dataset = CIFAR100(root="./data/CIFAR100", train=True, transform=context_transform, download=True)
    elif context_points == "cifar100":
        context_dataset = CIFAR10(root="./data/CIFAR10", train=True, transform=context_transform, download=True)
    elif context_points == "svhn":
        context_dataset = SVHN(root="./data/SVHN", split="train", download=True, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", transform=context_transform)
    else:
        ValueError("Unknown context dataset")
    
    full_context_dataset_size = len(context_dataset)
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, full_context_dataset_size - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)

    if ood_points == "svhn":
        ood_dataset = SVHN(root="./data/SVHN", split="test", download=True, transform=test_transform)
    elif ood_points == "cifar10":
        ood_dataset = CIFAR10(root="./data/CIFAR10", train=False, download=True, transform=test_transform)
    else:
        ValueError("Unknown OOD dataset")

    ood_dataset = CustomDataset(ood_dataset, len(test_dataset))
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

elif dataset == 'fmnist' or dataset == 'fmnist-224':
    _train_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, download=True)
    input_dim = 28
    num_classes = 10
    dataset_size = 60000
    testset_size = 10000
    if training_dataset_size == 0:
        training_dataset_size = 60000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2)).numpy()
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2)).numpy()
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)[:,:,None]
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img


    test_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    train_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    if dataset == 'fmnist-224':
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img

        test_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]
        train_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]

        test_transform_list.append(image_to_numpy)
        train_transform_list.append(image_to_numpy)

    _train_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=train_transform, download=True)
    val_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=test_transform, download=True)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = CustomDataset(train_dataset, dataset_size)
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = FashionMNIST(root="./data/fashionMNIST", train=False, transform=test_transform, download=True)

    if context_transform:
        context_transform_list = [
            transforms.Grayscale(1),
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(28),
        ]
        if dataset == 'fmnist-224':
            context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "train":
        context_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=context_transform, download=True)
    elif context_points == "kmnist":
        context_dataset = KMNIST(root="./data/", train=True, transform=context_transform, download=True)
    elif context_points == "mnist":
        context_dataset = MNIST("./data/", train=True, download=True, transform=context_transform)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", train=True, transform=context_transform, download=True)
    else:
        ValueError("Unknown context dataset")
    
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, 60000 - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)

    if ood_points == "mnist":
        ood_dataset = MNIST("./data/", train=False, download=True, transform=test_transform)
    elif ood_points == "kmnist":
        ood_dataset = KMNIST(root="./data/", train=False, transform=test_transform, download=True)
    else:
        ValueError("Unknown OOD dataset")

    ood_dataset = CustomDataset(ood_dataset, len(test_dataset))
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

elif dataset == 'mnist' or dataset == 'mnist-224':
    _train_dataset = FashionMNIST(root="./data/", train=True, download=True)
    input_dim = 28
    num_classes = 10
    dataset_size = 60000
    testset_size = 10000
    if training_dataset_size == 0:
        training_dataset_size = 60000
    validation_dataset_size = dataset_size - training_dataset_size
    batch_size_test = batch_size

    DATA_MEANS = (_train_dataset.data / 255.0).mean(axis=(0,1,2)).numpy()
    DATA_STD = (_train_dataset.data / 255.0).std(axis=(0,1,2)).numpy()
    print("Data mean", DATA_MEANS)
    print("Data std", DATA_STD)

    def image_to_numpy(img):
        img = np.array(img, dtype=np.float32)[:,:,None]
        img = (img / 255. - DATA_MEANS) / DATA_STD
        return img

    test_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    train_transform = transforms.Compose([
        # transforms.Normalize((0.2861,), (0.3530,)),
        # transforms.ToTensor(),
        image_to_numpy
    ])

    if dataset == 'mnist-224':
        def image_to_numpy(img):
            img = np.array(img, dtype=np.float32)
            img = (img / 255. - DATA_MEANS) / DATA_STD
            return img

        test_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]
        train_transform_list = [
            transforms.Resize(224),
            transforms.Grayscale(3),
        ]

        test_transform_list.append(image_to_numpy)
        train_transform_list.append(image_to_numpy)

    _train_dataset = MNIST(root="./data/", train=True, transform=train_transform, download=True)
    val_dataset = MNIST(root="./data/", train=True, transform=test_transform, download=True)

    train_dataset, _ = torch.utils.data.random_split(_train_dataset, [training_dataset_size, dataset_size-training_dataset_size], generator=torch.Generator().manual_seed(seed))
    train_dataset = CustomDataset(train_dataset, dataset_size)
    _, validation_dataset = torch.utils.data.random_split(val_dataset, [dataset_size-validation_dataset_size, validation_dataset_size], generator=torch.Generator().manual_seed(seed))

    test_dataset = MNIST(root="./data/", train=False, transform=test_transform, download=True)

    if context_transform:
        context_transform_list = [
            transforms.Grayscale(1),
            transforms.RandomCrop(input_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.GaussianBlur(kernel_size=(3,3)),
            transforms.GaussianBlur(kernel_size=(3,3), sigma=0.1),
            transforms.RandomSolarize(threshold=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.Resize(28),
        ]
        if dataset == 'mnist-224':
            context_transform_list.append(transforms.Resize(224))
        context_transform_list.append(image_to_numpy)
        context_transform = transforms.Compose(context_transform_list)
    else:
        context_transform = test_transform

    if context_points == "train":
        context_dataset = MNIST("./data/", train=True, download=True, transform=context_transform)
    elif context_points == "kmnist":
        context_dataset = KMNIST(root="./data/", train=True, transform=context_transform, download=True)
    elif context_points == "fmnist":
        context_dataset = FashionMNIST(root="./data/fashionMNIST", train=True, transform=context_transform, download=True)
    elif context_points == "imagenet":
        context_dataset = ImageNet(root="./data/ImageNet", train=True, transform=context_transform, download=True)
    else:
        ValueError("Unknown context dataset")
    
    context_set, _ = torch.utils.data.random_split(context_dataset, [context_dataset_size, 60000 - context_dataset_size], generator=torch.Generator().manual_seed(seed))
    context_set = CustomDataset(context_set, training_dataset_size)

    if ood_points == "fmnist":
        ood_dataset = FashionMNIST(root="./data/fashionMNIST", train=False, transform=test_transform, download=True)
    elif ood_points == "kmnist":
        ood_dataset = KMNIST(root="./data/", train=False, transform=test_transform, download=True)
    else:
        ValueError("Unknown OOD dataset")

    ood_dataset = CustomDataset(ood_dataset, len(test_dataset))
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )

elif dataset == 'two-moons':
    Path.mkdir(Path('figures/two_moons'), parents=True, exist_ok=True)

    input_dim = 2
    num_classes = 2
    dataset_size = 10000
    training_dataset_size = 200
    batch_size_test = 32
    num_workers = 0
    persistent_workers = False

    x_train, y_train = sklearn_datasets.make_moons(
        n_samples=training_dataset_size, shuffle=True, noise=0.2, random_state=seed
    )
    y_train = np.array(y_train[:, None] == np.arange(2))

    context_set_size = 10000
    x_context_lim = float(re.search('x_context_lim_(.*)', context_points).group(1))

    x_context_min, x_context_max = (
        x_train[:, 0].min() - x_context_lim,
        x_train[:, 0].max() + x_context_lim,
    )
    y_context_min, y_context_max = (
        x_train[:, 1].min() - x_context_lim,
        x_train[:, 1].max() + x_context_lim,
    )

    h_context_x = (x_context_max - x_context_min) / context_set_size ** 0.5
    h_context_y = (y_context_max - y_context_min) / context_set_size ** 0.5

    xx_context, yy_context = np.meshgrid(
        np.arange(x_context_min, x_context_max, h_context_x), np.arange(y_context_min, y_context_max, h_context_y)
    )
    x_context = np.vstack((xx_context.reshape(-1), yy_context.reshape(-1))).T
    y_context = jnp.ones((x_context.shape[0], 2))

    h = 0.25
    test_lim = 1.5
    x_min, x_max = (
        x_train[:, 0].min() - test_lim,
        x_train[:, 0].max() + test_lim,
        )
    y_min, y_max = (
        x_train[:, 1].min() - test_lim,
        x_train[:, 1].max() + test_lim,
        )
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )
    _x_test = np.vstack((xx.reshape(-1), yy.reshape(-1))).T
    _y_test = jnp.ones((_x_test.shape[0], 2))

    permutation = np.random.permutation(_x_test.shape[0])
    x_test = _x_test[permutation]
    y_test = _y_test[permutation]

    permutation_inv = np.argsort(permutation)

    h_wide = 0.25
    test_lim_wide = 1.5
    # test_lim_wide = 4
    x_wide_min, x_wide_max = (
        x_train[:, 0].min() - test_lim_wide,
        x_train[:, 0].max() + test_lim_wide,
    )
    y_wide_min, y_wide_max = (
        x_train[:, 1].min() - test_lim_wide,
        x_train[:, 1].max() + test_lim_wide,
    )
    xx_wide, yy_wide = np.meshgrid(
        np.arange(x_wide_min, x_wide_max, h_wide), np.arange(y_wide_min, y_wide_max, h_wide)
    )
    x_test_wide = np.vstack((xx_wide.reshape(-1), yy_wide.reshape(-1))).T
    y_test_wide = jnp.ones((x_test_wide.shape[0], 2))

    testset_size = x_test.shape[0]

    train_dataset = TwoMoons(x_train.shape[0], x_train, y_train)
    train_dataset = CustomDataset(train_dataset, dataset_size)
    context_set = TwoMoons(x_context.shape[0], x_context, y_context)
    context_set = CustomDataset(context_set, dataset_size)
    validation_dataset = TwoMoons(x_train.shape[0], x_train, y_train)
    test_dataset = TwoMoons(x_test.shape[0], x_test, y_test)

    ood_dataset = context_set
    ood_loader = data.DataLoader(ood_dataset,
                                 batch_size=context_batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=num_workers_test,
                                 persistent_workers=persistent_workers_test
                                 )
else:
    raise ValueError("Dataset not found.")
    
if dataset == "two-moons":
    num_workers_train = 0
    num_workers_test = 0
    persistent_workers_train = False
    persistent_workers_test = False
else:
    num_workers_train = os.cpu_count()
    persistent_workers_train = True

if context_points == "imagenet" or dataset == "two-moons":
    num_workers_context = 0
    persistent_workers_context = False
else:
    num_workers_context = os.cpu_count()
    persistent_workers_context = True

print(f"num_workers_train: {num_workers_train}")
print(f"persistent_workers_context: {persistent_workers_context}")

train_loader = data.DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               drop_last=True,
                               collate_fn=numpy_collate,
                               num_workers=num_workers_train,
                               persistent_workers=persistent_workers_train,
                               )
context_loader  = data.DataLoader(context_set,
                               batch_size=context_batch_size,
                               shuffle=True,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=num_workers_context,
                               persistent_workers=persistent_workers_context,
                               )
val_loader   = data.DataLoader(validation_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=num_workers_test,
                               persistent_workers=persistent_workers_test
                               )
test_loader  = data.DataLoader(test_dataset,
                               batch_size=batch_size_test,
                               shuffle=False,
                               drop_last=False,
                               collate_fn=numpy_collate,
                               num_workers=num_workers_test,
                               persistent_workers=persistent_workers_test
                               )

class TrainState(train_state.TrainState):
    batch_stats: Any
    # params_logvar: Any


class TrainerModule:
    def __init__(self,
                 model_name : str,
                 model_class : nn.Module,
                 optimizer_name : str,
                 model_hparams : dict,
                 optimizer_hparams : dict,
                 objective_hparams : dict,
                 other_hparams: dict,
                 exmp_inputs : Any,
                 ):
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.optimizer_name = optimizer_name

        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.objective_hparams = objective_hparams
        self.other_hparams = other_hparams
        
        self.seed = other_hparams["seed"]
        self.num_epochs = other_hparams["num_epochs"]
        self.evaluate = other_hparams["evaluate"]
        self.restore_checkpoint = other_hparams["restore_checkpoint"]
        self.batch_stats_init_epochs = other_hparams["batch_stats_init_epochs"]
        self.mc_samples_llk = objective_hparams["mc_samples_llk"]
        self.mc_samples_reg = objective_hparams["mc_samples_reg"]
        self.mc_samples_eval = other_hparams["mc_samples_eval"]
        self.dataset = other_hparams["dataset"]
        self.training_dataset_size = objective_hparams["training_dataset_size"]
        self.batch_size = objective_hparams["batch_size"]
        self.n_batches_train = self.training_dataset_size / self.batch_size
        self.num_classes = self.model_hparams["num_classes"]
        self.stochastic = other_hparams["stochastic"]
        self.learning_rate = self.optimizer_hparams["learning_rate"]
        self.learning_rate_scale_logvar = self.optimizer_hparams["learning_rate_scale_logvar"]
        self.alpha = self.optimizer_hparams["alpha"]
        self.weight_decay = self.optimizer_hparams["weight_decay"]
        self.momentum = self.optimizer_hparams['momentum']

        self.prior_mean = objective_hparams["prior_mean"]
        self.prior_var = objective_hparams["prior_var"]
        self.prior_likelihood_scale = objective_hparams["prior_likelihood_scale"]
        self.prior_likelihood_f_scale = objective_hparams["prior_likelihood_f_scale"]
        self.prior_likelihood_cov_scale = objective_hparams["prior_likelihood_cov_scale"]
        self.prior_likelihood_cov_diag = objective_hparams["prior_likelihood_cov_diag"]
        self.likelihood_scale = objective_hparams["likelihood_scale"]
        self.reg_scale = objective_hparams["reg_scale"]
        self.reg_type = self.objective_hparams["reg_type"]
        self.init_logvar = objective_hparams["init_logvar"]
        self.init_final_layer_weights_logvar = objective_hparams["init_final_layer_weights_logvar"]
        self.init_final_layer_bias_logvar = objective_hparams["init_final_layer_bias_logvar"]
        self.prior_feature_logvar = objective_hparams["prior_feature_logvar"]
        self.pretrained_prior = objective_hparams["pretrained_prior"]
        self.output_var = other_hparams["output_var"]
        
        self.prediction_type = other_hparams["prediction_type"]
        self.debug_print = other_hparams["debug_print"]
        self.debug_print_updated = other_hparams["debug_print"]
        self.log_frequency = other_hparams["log_frequency"]
        self.full_eval = other_hparams["full_eval"]
        self.save_to_wandb = other_hparams["save_to_wandb"]
        self.wandb_project = other_hparams["wandb_project"]
        self.wandb_account = other_hparams["wandb_account"]

        self.n_batches_eval = 100
        self.n_batches_eval_context = 10
        self.n_batches_eval_final = 100

        self.params_prior_mean = None
        self.params_prior_logvar = None
        self.batch_stats_prior = None
        self.pred_fn = None

        self.split_params = split_params

        self.run_name = f"{method}_{reg_type}_{prior_var}_{prior_mean}_{reg_scale}_{learning_rate}_{alpha}_{num_epochs}_{context_points}_{model_name}_{dataset}_{seed}"
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.run_name)

        self.logger = {
            "epoch": [],
            "loss_train": [],
            "acc_train": [],
            "acc_test": [],
            "acc_test_best": [],
            "acc_sel_test": [],
            "acc_sel_test_ood": [],
            "nll_test": [],
            "ece_test": [],
            "ood_auroc_entropy": [],
            "ood_auroc_aleatoric": [],
            "ood_auroc_epistemic": [],
            "predictive_entropy_test": [],
            "aleatoric_uncertainty_test": [],
            "epistemic_uncertainty_test": [],
            "predictive_entropy_context": [],
            "aleatoric_uncertainty_context": [],
            "epistemic_uncertainty_context": [],
            "predictive_entropy_ood": [],
            "aleatoric_uncertainty_ood": [],
            "epistemic_uncertainty_ood": [],
        }
        
        if "cifar10" in self.dataset and "cifar100" not in self.dataset:
            self.logger["acc_test_cifar101"] = []
            self.logger["acc_sel_test_cifar101"] = []
            self.logger["nll_test_cifar101"] = []
            self.logger["ece_test_cifar101"] = []
        if self.full_eval:
            if "cifar10" in self.dataset and "cifar100" not in self.dataset:
                for corr_config in corr_config_list:
                    self.logger[f"acc_test_ccifar10_{corr_config}"] = []
                    self.logger[f"acc_test_ccifar10_{corr_config}"] = []
                    self.logger[f"acc_sel_test_ccifar10_{corr_config}"] = []
        self.wandb_logger = []

        self.create_functions()
    
        if model_name == 'ResNet18-Pretrained':
            self.model = ResNet18(output='logits', pretrained='imagenet', \
                                    num_classes=self.num_classes, dtype='float32')
        elif model_name == 'ResNet50-Pretrained':
            self.model = ResNet50(output='logits', pretrained='imagenet', \
                                    num_classes=self.num_classes, dtype='float32')
        else:
            self.model = self.model_class(**self.model_hparams)        
        self.init_model(exmp_inputs)
        # print(self.model.tabulate(random.PRNGKey(0), x=exmp_inputs[0]))

        assert self.mc_samples_llk == 1 if not self.stochastic else True
        assert self.mc_samples_eval == 1 if not self.stochastic else True
        assert self.mc_samples_reg == 1 # if not ("fsmap" in method or "fsvi" in method) else True  # currently not implemented
        # assert self.objective_hparams["reg_points"] == "train" if self.objective_hparams["method"] == "psmap" else True

    def create_functions(self):
        def calculate_empirical_gaussian_prior_density(preds_f, params, inputs, batch_stats, prior_likelihood_cov_scale, prior_likelihood_cov_diag, rng_key):
            ### set prior batch stats
            if self.batch_stats_init_epochs == 0:
                batch_stats_prior = jax.lax.stop_gradient(batch_stats)
            else:
                batch_stats_prior = self.batch_stats_prior

            ### set parameter prior mean
            if self.params_prior_mean is not None:
                params_prior_mean = jax.lax.stop_gradient(self.params_prior_mean)
            else:
                params_prior_mean = jax.lax.stop_gradient(self.model.init(jax.random.PRNGKey(self.seed), inputs[0:1], train=True)["params"])
            params_feature_prior_mean, params_final_layer_prior_mean = self.split_params(params_prior_mean, "dense")

            ### initialize and split prior logvar parameters into feature and final-layer parameters
            params_prior_logvar_init = jax.tree_map(lambda x: x * 0, params_prior_mean)  # initialize logvar parameters with zeros
            params_feature_prior_logvar_init, params_final_layer_prior_logvar_init = self.split_params(params_prior_logvar_init, "dense")

            ### set feature parameter logvar and final-layer parameter variance
            feature_prior_logvar = self.prior_feature_logvar
            final_layer_prior_logvar = jnp.log(self.prior_var)
            params_feature_prior_logvar = jax.tree_map(lambda x: x * 0 + feature_prior_logvar, params_feature_prior_logvar_init)
            params_final_layer_prior_logvar = jax.tree_map(lambda x: x * 0 + final_layer_prior_logvar, params_final_layer_prior_logvar_init)

            params_feature_prior_sample = sample_parameters(params_feature_prior_mean, params_feature_prior_logvar, self.stochastic, rng_key)

            params_prior_sample = merge_params(params_feature_prior_sample, params_final_layer_prior_mean)

            ### feature covariance (mostly the same as Jacobian covariance, up to numerical errors)
            _out = self.model.apply({'params': params_prior_sample, 'batch_stats': batch_stats_prior},
                                    inputs,
                                    train=True,
                                    feature=True,
                                    mutable=['batch_stats'])
            out, _ = _out
            preds_f_prior_mean = jnp.zeros_like(jax.lax.stop_gradient(out[0]))
            feature_prior = jax.lax.stop_gradient(out[1])
            # preds_f_prior_mean, feature_prior = jax.lax.stop_gradient(out[0]), jax.lax.stop_gradient(out[1])

            preds_f_prior_cov = prior_likelihood_cov_scale * jnp.matmul(feature_prior, feature_prior.T)  # assumes the prior is identical across output dimensions
            preds_f_prior_cov += jnp.ones_like(preds_f_prior_cov) * prior_likelihood_cov_scale  # add bias variance
            preds_f_prior_cov += jnp.eye(preds_f_prior_cov.shape[0]) * prior_likelihood_cov_diag  # jnp.max(jnp.array([eps_cov * prior_var, eps_cov]))  # add small constant to the diagonal to ensure positive definiteness

            p = tfd.MultivariateNormalFullCovariance(
                loc=preds_f_prior_mean[:, 0],  # assumes the prior is identical across output dimensions
                covariance_matrix=preds_f_prior_cov,
                validate_args=False,
                allow_nan_stats=True,
            )
            log_density = jnp.sum(p.log_prob(preds_f[0].T))

            reg = -log_density
            
            ### L2 regularization on non-final-layer parameters
            reg += 1 / (2 * (self.prior_var)) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x: jnp.square(x), params))[0])  # 1/2 * ||params - params_prior_mean||^2

            if self.debug_print_updated:
                jax.debug.print("\nf - mean: {}", jnp.mean(preds_f[:, 0] - preds_f_prior_mean[:, 0]))
                jax.debug.print("log_density: {}\n", p.log_prob(preds_f.T))
                jax.debug.print("cholesky: {}\n", jnp.linalg.cholesky(preds_f_prior_cov[:, 0, :, 0]))

            return reg

        def calculate_parameter_norm(params, prior_var):
            params_model = params
            # params_model, params_batchnorm = self.split_params(params, "batch_norm")

            if self.objective_hparams["reg_type"] != "feature_parameter_norm":
                params_reg = params_model
                if self.params_prior_mean is None:
                    params_reg_prior_mean = jax.tree_map(lambda x: x * 0, jax.lax.stop_gradient(params_model))
                else:
                    params_reg_prior_mean = self.params_prior_mean
                reg = 1 / (2 * prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params - params_prior_mean||^2
            else:
                params_reg, _ = self.split_params(params_model, "dense")
                params_reg_prior_mean, _ = self.split_params(self.params_prior_mean, "dense")
                reg = 1 / (2 * prior_var) * jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(lambda x, y: jnp.square(x - y), params_reg, params_reg_prior_mean))[0])  # 1/2 * ||params_feature - params_feature_prior_mean||^2

            return reg  # this scaling makes prior_precision consistent with the MAP objective scaling but inconsistent with the weight decay coefficient

        def kl_univariate_gaussians(mean_q, var_q, mean_p, var_p):
            logstd_jitter = 0
            kl_1 = jnp.log((var_p + logstd_jitter) ** 0.5) - jnp.log((var_q + logstd_jitter) ** 0.5)
            kl_2 = ((var_q + logstd_jitter) + (mean_q - mean_p) ** 2) / (var_p + logstd_jitter)
            kl_3 = -1
            kl = 0.5 * (kl_1 + kl_2 + kl_3)

            return kl

        def calculate_parameter_kl(params_variational_mean, params_variational_logvar):
            if self.params_prior_mean is not None:
                params_prior_mean = self.params_prior_mean
                params_prior_logvar = self.params_prior_logvar
            else:
                params_prior_mean = jax.tree_map(lambda x: x * 0 + self.prior_mean, jax.lax.stop_gradient(params_variational_mean))
                params_prior_logvar = jax.tree_map(lambda x: x * 0 + jnp.log(self.prior_var), jax.lax.stop_gradient(params_variational_logvar))

            params_prior_var = jax.tree_map(lambda x: jnp.exp(x), params_prior_logvar)
            params_variational_var = jax.tree_map(lambda x: jnp.exp(x), params_variational_logvar)

            kl = jnp.sum(jax.flatten_util.ravel_pytree(jax.tree_map(
                lambda a, b, c, d: kl_univariate_gaussians(a, b, c, d),
                params_variational_mean, params_variational_var, params_prior_mean, params_prior_var
                ))[0])
            
            return kl

        def sample_parameters(params, params_logvar, stochastic, rng_key):
            params_random_mean = params
            params_random_logvar = params_logvar
                
            if stochastic:
                eps = jax.tree_map(lambda x: random.normal(rng_key, x.shape), params_random_logvar)
                params_std_sample = jax.tree_map(lambda x, y: x * jnp.exp(y) ** 0.5, eps, params_random_logvar)
                params_sample = jax.tree_map(lambda x, y: x + y, params_random_mean, params_std_sample)
            else:
                params_sample = params

            return params_sample

        def calculate_forward_pass(params, params_logvar, rng_key, batch_stats, inputs, _inputs_context, train):
            self.pred_fn = self.model.apply

            preds_f_llk_list = []
            preds_f_reg_list = []
            params_samples = []
            for _ in range(self.mc_samples_llk):
                rng_key, _ = jax.random.split(rng_key)

                if self.objective_hparams["stochastic"]:
                    params = sample_parameters(params, params_logvar, self.stochastic, rng_key)
                    params_samples.append(params)

                if (
                    self.objective_hparams["forward_points"] == "train" and
                    self.objective_hparams["reg_points"] == "train"
                    ):
                    inputs_forward = inputs  # inputs used to update batch stats
                    inputs_reg = inputs_forward

                    # a forward pass on the training points for the log-likelihood and regularization terms (batch stats are updated)
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs_forward,
                        train=train,
                        mutable=['batch_stats'] if train else False
                        )
                    preds_f_llk, new_model_state = out if train else (out, None)

                    preds_f_reg = preds_f_llk

                elif (
                    self.objective_hparams["forward_points"] == "train" and
                    self.objective_hparams["reg_points"] != "train"
                    ):
                    inputs_forward = inputs  # inputs used to update batch stats

                    # a forward pass on the training points for the log-likelihood term (batch stats are updated)
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs_forward,
                        train=train,
                        mutable=['batch_stats'] if train else False
                        )
                    preds_f_llk, new_model_state = out if train else (out, None)

                    if self.objective_hparams["reg_points"] == "context":
                        inputs_reg = _inputs_context

                        # a forward pass on the context points for the regularization term (batch stats are not updated)
                        out = self.pred_fn(
                            {'params': params, 'batch_stats': batch_stats},
                            inputs_reg,
                            train=train,
                            mutable=['batch_stats'] if train else False
                            )
                        _preds_f_reg, _ = out if train else (out, None)

                        preds_f_reg = jnp.concatenate([preds_f_llk, _preds_f_reg], axis=0)

                    elif self.objective_hparams["reg_points"] == "joint":
                        inputs_reg = jnp.concatenate([inputs, _inputs_context], axis=0)

                        # a forward pass on the joint points (context + train) for the regularization term (batch stats are not updated)
                        out = self.pred_fn(
                            {'params': params, 'batch_stats': batch_stats},
                            inputs_reg,
                            train=train,
                            mutable=['batch_stats'] if train else False
                            )
                        preds_f_reg, _ = out if train else (out, None)
                    else:
                        raise ValueError("Unknown forward_points/reg_points/context_points combination.")

                elif self.objective_hparams["forward_points"] == "joint":
                    inputs_forward = jnp.concatenate([inputs, _inputs_context], axis=0)  # inputs used to update batch stats

                    # a forward pass on both training and context points (batch stats are updated)
                    out = self.pred_fn(
                        {'params': params, 'batch_stats': batch_stats},
                        inputs_forward,
                        train=train,
                        mutable=['batch_stats'] if train else False
                        )
                    preds_f_joint, new_model_state = out if train else (out, None)

                    preds_f_llk = preds_f_joint[:inputs.shape[0]]

                    if self.objective_hparams["reg_points"] == "context":
                        inputs_reg = _inputs_context
                        preds_f_reg = preds_f_joint[-inputs_reg.shape[0]:]

                    elif self.objective_hparams["reg_points"] == "joint":
                        inputs_reg = jnp.concatenate([inputs, _inputs_context], axis=0)
                        preds_f_reg = preds_f_joint

                    else:
                        raise ValueError("Unknown forward_points/reg_points/context_points combination.")

                else:
                    raise ValueError("Unknown forward_points/reg_points/context_points combination.")
                    
                preds_f_llk_list.append(preds_f_llk)
                preds_f_reg_list.append(preds_f_reg)

            preds_f_llk = jnp.stack(preds_f_llk_list, axis=0)
            preds_f_reg = jnp.stack(preds_f_reg_list, axis=0)

            return preds_f_llk, preds_f_reg, params_samples, new_model_state, inputs_reg

        def calculate_loss(params, params_logvar, rng_key, batch_stats, batch, batch_context, train):
            inputs, targets = batch
            _inputs_context, _ = batch_context

            preds_f_llk, preds_f_reg, params_samples, new_model_state, inputs_reg = calculate_forward_pass(params, params_logvar, rng_key, batch_stats, inputs, _inputs_context, train)

            if self.reg_type == "parameter_kl":
                assert self.objective_hparams["stochastic"] == True
                reg = calculate_parameter_kl(params, params_logvar)
            elif self.reg_type == "empirical_gaussian_prior_density":
                reg = calculate_empirical_gaussian_prior_density(preds_f_reg, params, inputs_reg, batch_stats, self.prior_likelihood_cov_scale, self.prior_likelihood_cov_diag, rng_key)
            elif self.reg_type == "parameter_norm":
                reg = calculate_parameter_norm(params, self.prior_var)
            else:
                raise ValueError("Unknown regularization type.")

            scale = 1 / self.n_batches_train  # 1 / (number of mini-batches)
            reg = scale * reg

            if self.output_var:
                likelihood_scale = preds_f_llk[:, :, self.num_classes:]
                preds_f_llk = preds_f_llk[:, :, :self.num_classes]
            else:
                likelihood_scale = self.likelihood_scale

            if self.prediction_type == "classification":
                nll = categorical_nll_with_softmax(jax.nn.softmax((1 / likelihood_scale) * preds_f_llk, -1), targets).mean(0).sum()  # likelihood_scale = temperature
            elif self.prediction_type == "regression":
                nll = gaussian_nll(preds_f_llk, targets, likelihood_scale).mean(0).sum()  # likelihood_scale = likelihood variance
            else:
                raise ValueError("Unknown prediction type.")

            loss = (nll + self.reg_scale * reg) / self.batch_size  # per data point loss
            
            if self.prediction_type == "classification":
                acc = 100 * (preds_f_llk.argmax(axis=-1) == targets).mean()
            elif self.prediction_type == "regression":
                acc = jnp.mean(jnp.sum(jnp.square(jnp.mean(preds_f_llk, axis=0) - targets), axis=-1), axis=-1)
            else:
                raise ValueError("Unknown prediction type.")

            if self.debug_print_updated:
                jax.debug.print("nll: {}", nll)
                jax.debug.print("reg: {}", reg)
                jax.debug.print("loss: {}", loss)
                jax.debug.print("acc: {}", acc)

            return loss, (acc, new_model_state)

        @partial(jit, static_argnums=(5,))
        def pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key, feature=False):
            params = sample_parameters(params, params_logvar, self.stochastic, rng_key)

            if feature:
                preds_f, feature = self.pred_fn(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=False,
                    mutable=False,
                    feature=feature
                    )
                return preds_f, feature        
            else:
                preds_f = self.pred_fn(
                    {'params': params, 'batch_stats': batch_stats},
                    inputs,
                    train=False,
                    mutable=False
                    )
                return preds_f

        def evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, type):
            if type == "test":
                _logits_test = []
                _targets_test = []

                for i, batch in enumerate(test_loader):
                    inputs_test, _targets = batch
                    inputs = inputs_test
                    _logits_test_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_test_list.append(pred)
                    _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                    _targets_test.append(_targets)
                    if i == n_batches_eval:
                        break
                logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                targets_test = jnp.concatenate(_targets_test, axis=0)

                ret = [logits_test, targets_test]

            elif type == "context":
                _logits_context = []
                for i, batch in enumerate(context_loader):
                    inputs_context, _ = batch
                    n_context_points = inputs_context.shape[0]
                    inputs = jnp.concatenate([inputs_context, inputs_context], axis=0)
                    _logits_context_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        _pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        pred = _pred[:_pred.shape[0] - n_context_points]
                        _logits_context_list.append(pred)
                    _logits_context.append(jnp.stack(_logits_context_list, axis=0))
                    if i == self.n_batches_eval_context:
                        break
                logits_context = jnp.concatenate(_logits_context, axis=1)

                ret = [logits_context, None]

            elif type == "ood":
                _logits_ood = []
                for i, batch in enumerate(ood_loader):
                    inputs_ood, _ = batch
                    inputs = inputs_ood
                    _logits_ood_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_ood_list.append(pred)
                    _logits_ood.append(jnp.stack(_logits_ood_list, axis=0))
                    if i == n_batches_eval:
                        break
                logits_ood = jnp.concatenate(_logits_ood, axis=1)

                ret = [logits_ood, None]

            if type == "cifar101":
                _logits_test = []
                _targets_test = []

                for i, batch in enumerate(cifar101test_loader):
                    inputs_test, _targets = batch
                    inputs_test, _targets = jnp.array(inputs_test), jnp.array(_targets)
                    inputs = inputs_test
                    _logits_test_list = []
                    for _ in range(self.mc_samples_eval):
                        rng_key, _ = jax.random.split(rng_key)
                        pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                        _logits_test_list.append(pred)
                    _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                    _targets_test.append(_targets)
                    if i == n_batches_eval:
                        break
                logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                targets_test = jnp.concatenate(_targets_test, axis=0)

                ret = [logits_test, targets_test]

            if type == "corruptedcifar10":
                _logits_test_list_full = []
                _targets_test_list_full = []

                for ccifar10test_loader in ccifar10test_loader_list:
                    _logits_test = []
                    _targets_test = []
                    for i, batch in enumerate(ccifar10test_loader):
                        inputs_test, _targets = batch
                        inputs_test, _targets = jnp.array(inputs_test), jnp.array(_targets)
                        inputs = inputs_test
                        _logits_test_list = []
                        for _ in range(self.mc_samples_eval):
                            rng_key, _ = jax.random.split(rng_key)
                            pred = pred_fn_jit(params, params_logvar, batch_stats, inputs, rng_key)
                            _logits_test_list.append(pred)
                        _logits_test.append(jnp.stack(_logits_test_list, axis=0))
                        _targets_test.append(_targets)
                        if i == n_batches_eval:
                            break
                    logits_test = jnp.concatenate(_logits_test, axis=1)[:, :testset_size, :]
                    targets_test = jnp.concatenate(_targets_test, axis=0)[:testset_size]
                    _logits_test_list_full.append(logits_test)
                    _targets_test_list_full.append(targets_test)

                ret = [_logits_test_list_full, _targets_test_list_full]

            return ret

        def calculate_metrics(params, params_logvar, rng_key, batch_stats, n_batches_eval, full_eval):
            logits_test, targets_test = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "test")
            assert not np.any(np.isnan(logits_test)), f"logits_test contains NaNs."

            logits_context, _ = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "context")
            assert not np.any(np.isnan(logits_context)), f"logits_context contains NaNs."

            logits_ood, _ = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "ood")
            assert not np.any(np.isnan(logits_ood)), f"logits_ood contains NaNs."

            if "cifar10" in self.dataset and "cifar100" not in self.dataset:
                logits_cifar101, targets_cifar101 = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "cifar101")

                acc_test_cifar101 = 100 * np.array(np.mean(jax.nn.softmax(logits_cifar101, axis=-1).mean(0).argmax(axis=-1) == targets_cifar101))
                acc_sel_test_cifar101 = selective_accuracy(jax.nn.softmax(logits_cifar101, axis=-1).mean(0), targets_cifar101)
                nll_test_cifar101 = float(categorical_nll_with_softmax(jax.nn.softmax(logits_cifar101, -1).mean(0), targets_cifar101).mean())
                ece_test_cifar101 = 100 * calibration(jax.nn.one_hot(targets_cifar101, self.num_classes), jax.nn.softmax(logits_cifar101, axis=-1).mean(0))[0]

                self.logger["acc_test_cifar101"].append(acc_test_cifar101)
                self.logger["acc_sel_test_cifar101"].append(acc_sel_test_cifar101)
                self.logger["nll_test_cifar101"].append(nll_test_cifar101)
                self.logger["ece_test_cifar101"].append(ece_test_cifar101)

            if full_eval:
                if "cifar10" in self.dataset and "cifar100" not in self.dataset:
                    logits_ccifar10_list, targets_ccifar10_list = self.evaluation_predictions(params, params_logvar, rng_key, batch_stats, n_batches_eval, "corruptedcifar10")

                    for i, corr_config in enumerate(corr_config_list):
                        acc_test_ccifar10 = 100 * np.array(np.mean(jax.nn.softmax(logits_ccifar10_list[i], axis=-1).mean(0).argmax(axis=-1) == targets_ccifar10_list[i]))
                        acc_sel_test_ccifar10 = selective_accuracy(jax.nn.softmax(logits_ccifar10_list[i], axis=-1).mean(0), targets_ccifar10_list[i])
                        self.logger[f"acc_test_ccifar10_{corr_config}"].append(acc_test_ccifar10)
                        self.logger[f"acc_sel_test_ccifar10_{corr_config}"].append(acc_sel_test_ccifar10)

            acc_test = 100 * np.array(np.mean(jax.nn.softmax(logits_test, axis=-1).mean(0).argmax(axis=-1) == targets_test))
            acc_sel_test = selective_accuracy(jax.nn.softmax(logits_test, axis=-1).mean(0), targets_test)
            acc_sel_test_ood = selective_accuracy_test_ood(
                jax.nn.softmax(logits_test, axis=-1).mean(0),
                jax.nn.softmax(logits_ood, axis=-1).mean(0),
                targets_test
                )
            nll_test = float(categorical_nll_with_softmax(jax.nn.softmax(logits_test, -1).mean(0), targets_test).mean())
            ece_test = 100 * calibration(jax.nn.one_hot(targets_test, self.num_classes), jax.nn.softmax(logits_test, axis=-1).mean(0))[0]

            ood_auroc_entropy = 100 * auroc_logits(logits_test, logits_ood, score="entropy", rng_key=rng_key)
            ood_auroc_aleatoric = 100 * auroc_logits(logits_test, logits_ood, score="expected entropy", rng_key=rng_key)
            ood_auroc_epistemic = 100 * auroc_logits(logits_test, logits_ood, score="mutual information", rng_key=rng_key)

            predictive_entropy_test = float(categorical_entropy(jax.nn.softmax(logits_test, -1).mean(0)).mean(0))
            predictive_entropy_context = float(categorical_entropy(jax.nn.softmax(logits_context, -1).mean(0)).mean(0))
            predictive_entropy_ood = float(categorical_entropy(jax.nn.softmax(logits_ood, -1).mean(0)).mean(0))
            aleatoric_uncertainty_test = float(categorical_entropy(jax.nn.softmax(logits_test, -1)).mean(0).mean(0))
            aleatoric_uncertainty_context = float(categorical_entropy(jax.nn.softmax(logits_context, -1)).mean(0).mean(0))
            aleatoric_uncertainty_ood = float(categorical_entropy(jax.nn.softmax(logits_ood, -1)).mean(0).mean(0))
            epistemic_uncertainty_test = predictive_entropy_test - aleatoric_uncertainty_test
            epistemic_uncertainty_context = predictive_entropy_context - aleatoric_uncertainty_context
            epistemic_uncertainty_ood = predictive_entropy_ood - aleatoric_uncertainty_ood

            self.logger["acc_test"].append(acc_test)
            self.logger["acc_sel_test"].append(acc_sel_test)
            self.logger["acc_sel_test_ood"].append(acc_sel_test_ood)
            self.logger["nll_test"].append(nll_test)
            self.logger["ece_test"].append(ece_test)
            self.logger["ood_auroc_entropy"].append(ood_auroc_entropy)
            self.logger["ood_auroc_aleatoric"].append(ood_auroc_aleatoric)
            self.logger["ood_auroc_epistemic"].append(ood_auroc_epistemic)
            self.logger["predictive_entropy_test"].append(predictive_entropy_test)
            self.logger["predictive_entropy_context"].append(predictive_entropy_context)
            self.logger["predictive_entropy_ood"].append(predictive_entropy_ood)
            self.logger["aleatoric_uncertainty_test"].append(aleatoric_uncertainty_test)
            self.logger["aleatoric_uncertainty_context"].append(aleatoric_uncertainty_context)
            self.logger["aleatoric_uncertainty_ood"].append(aleatoric_uncertainty_ood)
            self.logger["epistemic_uncertainty_test"].append(epistemic_uncertainty_test)
            self.logger["epistemic_uncertainty_context"].append(epistemic_uncertainty_context)
            self.logger["epistemic_uncertainty_ood"].append(epistemic_uncertainty_ood)

        @partial(jit, static_argnums=(4,))
        def train_step(state, batch, batch_context, rng_key, debug_print):
            self.debug_print_updated = debug_print

            params = state.params["params"]
            params_logvar = state.params["params_logvar"]

            loss_fn = lambda params, params_logvar: calculate_loss(params, params_logvar, rng_key, state.batch_stats, batch, batch_context, train=True)
            # Get loss, gradients for loss, and other outputs of loss function
            ret, _grads = jax.value_and_grad(loss_fn, argnums=(0,1,), has_aux=True)(params, params_logvar)
            grads, grads_logvar = _grads[0], jax.tree_map(lambda x: self.learning_rate_scale_logvar * x, _grads[1])
            loss, acc, new_model_state = ret[0], *ret[1]
            # Update parameters and batch statistics
            state = state.apply_gradients(grads=freeze({"params": grads, "params_logvar": grads_logvar}), batch_stats=new_model_state['batch_stats'])
            return state, loss, acc

        def eval_step(state, rng_key, n_batches_eval, full_eval):
            calculate_metrics(state.params["params"], state.params["params_logvar"], rng_key, state.batch_stats, n_batches_eval, full_eval)
    
        self.train_step = train_step
        self.evaluation_predictions = evaluation_predictions
        self.eval_step = eval_step
        self.pred_fn_jit = pred_fn_jit

    def init_model(self, exmp_inputs):
        init_rng = jax.random.PRNGKey(self.seed)
        init_rng_logvar, _ = random.split(init_rng)

        self.optimizer_hparams.pop("learning_rate")
        self.optimizer_hparams.pop("learning_rate_scale_logvar")
        self.optimizer_hparams.pop("alpha")

        variables = self.model.init(init_rng, exmp_inputs, train=True)
        variables_logvar = self.model.init(init_rng_logvar, exmp_inputs, train=True)

        init_params = variables['params']

        if self.stochastic:
            init_params_logvar = jax.tree_map(lambda x: x + self.init_logvar, variables_logvar['params'])
            init_params_feature_logvar, init_params_final_layer_logvar = self.split_params(init_params_logvar, "dense")
            init_params_final_layer_logvar = jax.tree_map(lambda x: x * 0 + self.init_logvar, init_params_final_layer_logvar)
            self.final_layer_key = [key for key in init_params_final_layer_logvar.keys()][-1]

            minval_weights = self.init_final_layer_weights_logvar
            maxval_weights = self.init_final_layer_weights_logvar + 0.1
            init_params_final_layer_logvar[self.final_layer_key]["kernel"] = jnp.array(jax.random.uniform(key=init_rng, shape=init_params_final_layer_logvar[self.final_layer_key]["kernel"].shape, minval=minval_weights, maxval=maxval_weights, dtype=float))

            minval_bias = self.init_final_layer_bias_logvar
            maxval_bias = self.init_final_layer_bias_logvar + 0.1
            init_params_final_layer_logvar[self.final_layer_key]["bias"] = jnp.array(jax.random.uniform(key=init_rng, shape=init_params_final_layer_logvar[self.final_layer_key]["bias"].shape, minval=minval_bias, maxval=maxval_bias, dtype=float))

            init_params_logvar = merge_params(init_params_feature_logvar, init_params_final_layer_logvar)
        else:
            init_params_logvar = None

        self.init_params = freeze({"params": init_params, "params_logvar": init_params_logvar})
        self.init_batch_stats = variables['batch_stats']

        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
            self.optimizer_hparams.pop('momentum')
            self.optimizer_hparams.pop('weight_decay')
        elif self.optimizer_name.lower() == 'adamw':
            self.optimizer_hparams.pop('momentum')
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{self.optimizer_name}"'

        if self.alpha != 1:
            learning_rate_schedule = optax.cosine_decay_schedule(
                init_value=self.learning_rate,
                decay_steps=num_steps_per_epoch*num_epochs,
                alpha=self.alpha,
            )
        else:
            learning_rate_schedule = optax.piecewise_constant_schedule(
                init_value=self.learning_rate
            )
        transf = []
        # transf = [optax.clip(1.0)]
        if (opt_class == optax.sgd or opt_class == optax.adamw) and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.weight_decay))
            self.optimizer_hparams.pop('weight_decay')
        optimizer = optax.chain(
            *transf,
            opt_class(learning_rate_schedule, **self.optimizer_hparams)
        )

        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
            tx=optimizer
            )

    def train_model(self, train_loader, context_loader, val_loader, rng_key, num_epochs=200):
        print(f"\nTraining model for {num_epochs} epochs:\n")
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval = 0.0

        if self.batch_stats_prior is None and self.batch_stats_init_epochs != 0:
            print(f"Calibrating batch normalization statistics for {self.batch_stats_init_epochs} epochs:\n")
            self.batch_stats_prior = self.state.batch_stats
            self.state_pretrain = self.state
            for _ in tqdm(range(self.batch_stats_init_epochs)):
                self.pretrain(train_loader, context_loader, rng_key=rng_key)
            self.state = self.state.replace(batch_stats=self.state_pretrain.batch_stats)
            self.batch_stats_prior = self.state_pretrain.batch_stats

        for epoch in tqdm(range(num_epochs), leave=False):
            epoch_idx = epoch + 1
            self.train_epoch(train_loader, context_loader, epoch=epoch_idx, rng_key=rng_key)
            if epoch_idx % self.log_frequency == 0:
                if self.dataset != "two-moons":
                    self.eval_model(rng_key, self.n_batches_eval)
                    if self.logger['acc_test'][-1] >= best_eval:
                        self.logger['acc_test_best'].append(self.logger['acc_test'][-1])
                        self.save_model(step=epoch_idx, best=True)
                    else:
                        self.logger['acc_test_best'].append(self.logger['acc_test_best'][-1])
                    best_eval = self.logger['acc_test_best'][-1]

                    self.logger['epoch'].append(epoch_idx) 

                    if self.save_to_wandb and epoch_idx < num_epochs:
                        self.wandb_logger.append({})
                        for item in self.logger.items():
                            try:
                                self.wandb_logger[-1][item[0]] = item[1][-1]
                            except:
                                pass
                        wandb.log(self.wandb_logger[-1])

                    if epoch_idx % (10 * self.log_frequency) == 0:
                        self.save_model(step=epoch_idx)

                    print(f"\nEpoch {epoch_idx}  |  Train Accuracy: {self.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {self.logger['acc_test'][-1]:.2f}  |  Selective Accuracy Test: {self.logger['acc_sel_test'][-1]:.2f}  |  Selective Accuracy Test+OOD: {self.logger['acc_sel_test_ood'][-1]:.2f}  |  NLL: {self.logger['nll_test'][-1]:.3f}  |  Test ECE: {self.logger['ece_test'][-1]:.2f}  |  OOD AUROC: {self.logger['ood_auroc_entropy'][-1]:.2f} / {self.logger['ood_auroc_aleatoric'][-1]:.2f} / {self.logger['ood_auroc_epistemic'][-1]:.2f}  |  Uncertainty Test: {self.logger['predictive_entropy_test'][-1]:.3f} / {self.logger['aleatoric_uncertainty_test'][-1]:.3f} / {self.logger['epistemic_uncertainty_test'][-1]:.3f}  |  Uncertainty Context: {self.logger['predictive_entropy_context'][-1]:.3f} / {self.logger['aleatoric_uncertainty_context'][-1]:.3f} / {self.logger['epistemic_uncertainty_context'][-1]:.3f}  |  Uncertainty OOD: {self.logger['predictive_entropy_ood'][-1]:.3f} / {self.logger['aleatoric_uncertainty_ood'][-1]:.3f} / {self.logger['epistemic_uncertainty_ood'][-1]:.3f}")

                elif self.dataset == "two-moons":  # two-moons                                     
                    _preds_f_test_samples = []
                    for i, batch in enumerate(test_loader):
                        inputs_test, _ = batch
                        inputs = inputs_test
                        _preds_f_test_samples_list = []
                        for _ in range(self.mc_samples_eval):
                            rng_key, _ = jax.random.split(rng_key)
                            sample = self.pred_fn_jit(self.state.params["params"], self.state.params["params_logvar"], self.state.batch_stats, inputs, rng_key)
                            _preds_f_test_samples_list.append(sample)
                        _preds_f_test_samples.append(jnp.stack(_preds_f_test_samples_list, axis=0))
                    preds_f_test_samples = jnp.concatenate(_preds_f_test_samples, axis=1)[:, :testset_size, :]

                    preds_y_mean = jnp.mean(jax.nn.softmax(preds_f_test_samples, -1), 0)
                    preds_y_mean = preds_y_mean[permutation_inv]
                    prediction_mean = preds_y_mean[:, 0].reshape(xx.shape)

                    preds_y_var = jnp.var(jax.nn.softmax(preds_f_test_samples, -1), 0)
                    preds_y_var = preds_y_var[permutation_inv]
                    prediction_var = preds_y_var[:, 0].reshape(xx.shape)

                    mc_samples_eval_plot = 10

                    preds_f_sample_train_list = []
                    for _ in range(mc_samples_eval_plot):
                        rng_key, _ = jax.random.split(rng_key)
                        sample = self.pred_fn_jit(self.state.params["params"], self.state.params["params_logvar"], self.state.batch_stats, x_train, rng_key)
                        preds_f_sample_train_list.append(sample)
                    preds_f_sample_train = jnp.stack(preds_f_sample_train_list, axis=0)
                    ece_train = 100 * calibration(jax.nn.one_hot(jnp.array(y_train[:, 0], dtype=int), 2), jnp.mean(jax.nn.softmax(preds_f_sample_train, -1), 0))[0]

                    print(f"ECE Train: {ece_train:.2f}")

                    plt.figure(figsize=(10, 7))
                    cbar = plt.contourf(xx, yy, prediction_mean, levels=20, cmap=cm.coolwarm)
                    cb = plt.colorbar(cbar,)
                    cb.ax.set_ylabel(
                        # "$\mathbb{E}_{\Theta}[y(x_{*}, \Theta) | \mathcal{D}]$",
                        "$\mathbb{E}_{\Theta}[p(y_{*} | x_{*}, \Theta; f) | \mathcal{D}]$",
                        rotation=270,
                        labelpad=40,
                        size=30,
                    )
                    # cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
                    cb.ax.tick_params(labelsize=30)
                    plt.scatter(
                        x_train[y_train[:, 0] == 1, 0],
                        x_train[y_train[:, 0] == 1, 1],
                        color="cornflowerblue",
                        edgecolors="black",
                        s=90
                    )
                    plt.scatter(
                        x_train[y_train[:, 0] == 0, 0],
                        x_train[y_train[:, 0] == 0, 1],
                        color="tomato",
                        edgecolors="black",
                        s=90
                    )
                    plt.tick_params(labelsize=30)
                    plt.xticks([])
                    plt.yticks([])
                    plt.savefig(
                        f"figures/two_moons/two_moons_predictive_mean_{epoch_idx}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()

                    plt.figure(figsize=(10, 7))
                    cbar = plt.contourf(xx, yy, prediction_var, levels=20, cmap=cm.coolwarm)
                    cb = plt.colorbar(cbar,)
                    cb.ax.set_ylabel(
                        "$\mathbb{V}_{\Theta}[y(x_{*}, \Theta) | \mathcal{D}]$",
                        rotation=270,
                        labelpad=40,
                        size=30,
                    )
                    # cb.ax.set_ylabel('$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$', labelpad=-90)
                    cb.ax.tick_params(labelsize=30)
                    plt.scatter(
                        x_train[y_train[:, 0] == 0, 0],
                        x_train[y_train[:, 0] == 0, 1],
                        color="cornflowerblue",
                        edgecolors="black",
                        s=90
                    )
                    plt.scatter(
                        x_train[y_train[:, 0] == 1, 0],
                        x_train[y_train[:, 0] == 1, 1],
                        color="tomato",
                        edgecolors="black",
                        s=90
                    )
                    plt.tick_params(labelsize=30)
                    plt.savefig(
                        f"figures/two_moons/two_moons_predictive_var_{epoch_idx}.pdf",
                        bbox_inches="tight",
                    )
                    plt.close()
                else:
                    raise ValueError(f"Dataset {self.dataset} not recognized.")

    def pretrain(self, train_loader, context_loader, rng_key):
        data_loader = tqdm(zip(train_loader, context_loader), leave=False)
        for batch, batch_context in data_loader:
            self.state_pretrain, loss, acc = self.train_step(self.state_pretrain, batch, batch_context, rng_key)
            rng_key, _ = jax.random.split(rng_key)
            self.batch_stats_prior = self.state_pretrain.batch_stats


    def train_epoch(self, train_loader, context_loader, epoch, rng_key):
        metrics = defaultdict(list)
        data_loader = tqdm(enumerate(zip(train_loader, context_loader)), leave=False)
        train_acc = 0
        elapsed = 0
        for i, (batch, batch_context) in data_loader:
            if self.debug_print:
                if i % 1000 == 0:
                    debug_print = True
                    print(f"\nEpoch {epoch} Batch {i}   \n")
                else:
                    debug_print = False
            else:
                debug_print = False
            self.state, loss, acc = self.train_step(self.state, batch, batch_context, rng_key, debug_print)
            rng_key, _ = jax.random.split(rng_key)
            metrics['loss'].append(loss)
            metrics['acc'].append(acc)
            if (data_loader.format_dict["elapsed"] - elapsed) >= 0.5:  # update every 5 seconds
                train_acc = np.stack(jax.device_get(metrics["acc"]))[-40:].mean()  # average accuracy of last 40 batches
                train_loss = np.stack(jax.device_get(metrics["loss"]))[-40:].mean()  # average accuracy of last 40 batches
                data_loader.set_postfix({'accuracy': train_acc, 'loss': train_loss})
                elapsed = data_loader.format_dict["elapsed"]
        for key in metrics:
            avg_val = np.stack(jax.device_get(metrics[key])).mean()
            self.logger[f"{key}_train"].append(avg_val)

    def eval_model(self, rng_key, n_batches_eval, full_eval=False):
        self.eval_step(self.state, rng_key, n_batches_eval, full_eval)

    def save_model(self, step=0, best=False):
        if best:
            checkpoints.save_checkpoint(
                ckpt_dir=f"{self.log_dir}_{best}",
                target={
                    'params': self.state.params["params"],
                    'params_logvar': self.state.params["params_logvar"],
                    'batch_stats': self.state.batch_stats,
                    'batch_stats_prior': self.batch_stats_prior
                },
                step=0,
                overwrite=True,
                prefix=f"checkpoint_best_"
                )
            if self.save_to_wandb:
                wandb.save(f'{self.log_dir}/checkpoint_best_0')
        else:
            checkpoints.save_checkpoint(
                ckpt_dir=f"{self.log_dir}_{step}",
                target={
                    'params': self.state.params["params"],
                    'params_logvar': self.state.params["params_logvar"],
                    'batch_stats': self.state.batch_stats,
                    'batch_stats_prior': self.batch_stats_prior
                },
                step=step,
                overwrite=True,
                prefix=f"checkpoint_"
                )
            if self.save_to_wandb:
                wandb.save(f'{self.log_dir}/checkpoint_{step}')

    def load_model(self, stochastic=False, pretrained_prior=False, restore_checkpoint=False):
        if not stochastic:
            if "Pretrained" in self.model_name and not restore_checkpoint:
                state_dict = self.model.init(rng_key, jnp.ones((1, 224, 224, 3)))
            else:
                ckpt_path = f"/scratch-ssd/timner/deployment/testing/projects/function-space-variational-inference/{self.log_dir}_50/checkpoint_50"

                state_dict = freeze(checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None))
            params_logvar = None
        else:
            if "Pretrained" in self.model_name and not restore_checkpoint:
                state_dict = self.model.init(rng_key, jnp.ones((1, 224, 224, 3)))

                params_logvar = jax.tree_map(lambda x: x + self.init_logvar, state_dict['params'])
                params_feature_logvar, params_final_layer_logvar = self.split_params(params_logvar, "dense")
                params_final_layer_logvar[self.final_layer_key]["bias"] = params_final_layer_logvar[self.final_layer_key]["bias"] * 0 + self.init_final_layer_bias_logvar
                params_logvar = merge_params(params_feature_logvar, params_final_layer_logvar)
            else:
                ckpt_path = None  # specify path to checkpoint
                state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_path, target=None)
                params_logvar = state_dict['params_logvar']

        if pretrained_prior:
            self.params_prior_mean = state_dict['params']
            self.batch_stats_prior = state_dict['batch_stats']
            self.params_prior_logvar = jax.tree_map(lambda x: x * 0 + jnp.log(self.prior_var), self.params_prior_mean)
        
        self.pred_fn = self.model.apply

        params = freeze({"params": state_dict['params'], "params_logvar": params_logvar})

        self.state = TrainState.create(apply_fn=self.model.apply,
                                    params=params,
                                    batch_stats=state_dict['batch_stats'],
                                    tx=self.state.tx if self.state else optax.sgd(0.1)   # Default optimizer
                                    )

    def checkpoint_exists(self):
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))


def trainer(*args, rng_key, **kwargs):
    kwargs_copy = deepcopy(kwargs)
    del kwargs_copy['exmp_inputs']

    trainer = TrainerModule(*args, **kwargs)

    pprint(kwargs_copy)
    if trainer.save_to_wandb:
        wandb.config = kwargs_copy
        wandb.init(
            project=trainer.wandb_project,
            name=trainer.run_name,
            entity=trainer.wandb_account,
            config=wandb.config,
        )

    train = not trainer.evaluate

    if train and not trainer.restore_checkpoint and not "Pretrained" in kwargs['model_name']:  # train from scratch
        trainer.train_model(train_loader, context_loader, val_loader, rng_key, num_epochs=trainer.num_epochs)
    elif train and ("Pretrained" in kwargs['model_name'] or trainer.restore_checkpoint):  # load trained model and continue training
        trainer.load_model(stochastic=trainer.stochastic, pretrained_prior=trainer.pretrained_prior, restore_checkpoint=trainer.restore_checkpoint)
        trainer.train_model(train_loader, context_loader, val_loader, rng_key, num_epochs=trainer.num_epochs)
    else:  # load trained model and evaluate
        trainer.load_model(stochastic=trainer.stochastic, pretrained_prior=trainer.pretrained_prior, restore_checkpoint=trainer.restore_checkpoint)
        trainer.logger['acc_train'].append(0)
        trainer.logger['acc_test_best'].append(0)
        trainer.logger['loss_train'].append(0)
        trainer.logger['epoch'].append(trainer.num_epochs)

    if trainer.dataset != "two-moons" and trainer.dataset != "snelson" and trainer.dataset != "oat1d" and "offline_rl" not in trainer.dataset:
        # val_acc = trainer.eval_model(val_loader, rng_key)
        trainer.eval_model(rng_key, trainer.n_batches_eval_final, full_eval=trainer.full_eval)
        # print(f"\nValidation Accuracy: {val_acc*100:.2f}")
        print(f"Train Accuracy: {trainer.logger['acc_train'][-1]:.2f}  |  Test Accuracy: {trainer.logger['acc_test'][-1]:.2f}  |  Selective Accuracy: {trainer.logger['acc_sel_test'][-1]:.2f}  |  Selective Accuracy Test+OOD: {trainer.logger['acc_sel_test_ood'][-1]:.2f}  |  NLL: {trainer.logger['nll_test'][-1]:.3f}  |  Test ECE: {trainer.logger['ece_test'][-1]:.2f}  |  OOD AUROC: {trainer.logger['ood_auroc_entropy'][-1]:.2f} / {trainer.logger['ood_auroc_aleatoric'][-1]:.2f} / {trainer.logger['ood_auroc_epistemic'][-1]:.2f}  |  Uncertainty Test: {trainer.logger['predictive_entropy_test'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_test'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_test'][-1]:.3f}  |  Uncertainty Context: {trainer.logger['predictive_entropy_context'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_context'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_context'][-1]:.3f}  |  Uncertainty OOD: {trainer.logger['predictive_entropy_ood'][-1]:.3f} / {trainer.logger['aleatoric_uncertainty_ood'][-1]:.3f} / {trainer.logger['epistemic_uncertainty_ood'][-1]:.3f}")
        
        trainer.wandb_logger.append({})
        for item in trainer.logger.items():
            trainer.wandb_logger[-1][item[0]] = item[1][-1]

        if trainer.save_to_wandb:
            wandb.log(trainer.wandb_logger[-1])
            time.sleep(10)
            
            pprint(trainer.wandb_logger[-1])
    
    return trainer, trainer.logger


# conv_kernel_init = lecun_normal()  # flax default
conv_kernel_init = nn.initializers.variance_scaling(1/20, mode='fan_in', distribution='uniform')
# conv_kernel_init = nn.initializers.variance_scaling(1/20, mode='fan_in', distribution='normal')
# conv_kernel_init = nn.initializers.variance_scaling(2.0, mode='fan_out', distribution='normal')


class CNN(nn.Module):
    """A simple CNN model."""
    num_classes : int
    act_fn : callable
    block_class : None
    num_blocks : None
    c_hidden : None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        x = nn.Conv(features=32, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = nn.Conv(features=64, kernel_size=(3, 3), kernel_init=conv_kernel_init, dtype=self.dtype)(x)
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256, dtype=self.dtype)(x)
        x = nn.relu(x)

        feature_map = x

        x = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        
        if feature:
            return (x, feature_map)
        else:
            return x


class MLP_Toy(nn.Module):
    """A simple MLP model."""
    num_classes : int
    act_fn : callable  # unused
    block_class : None  # unused
    num_blocks : None  # unused
    c_hidden : None  # unused
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        _ = nn.BatchNorm(dtype=self.dtype)(x, use_running_average=not train)
        x = nn.Dense(features=200, dtype=self.dtype)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=200, dtype=self.dtype)(x)
        x = nn.tanh(x)

        feature_map = x

        x = nn.Dense(features=num_classes, dtype=self.dtype)(x)
        
        if feature:
            return (x, feature_map)
        else:
            return x


class ResNetBlock(nn.Module):
    act_fn : callable  # Activation function
    c_out : int   # Output feature size
    subsample : bool = False  # If True, we apply a stride inside F

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=conv_kernel_init,
            use_bias=False
            )(x)
        z = nn.BatchNorm()(z, use_running_average=not train)

        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=conv_kernel_init,
            use_bias=False
            )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)

        if self.subsample:
            x = nn.Conv(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=conv_kernel_init
            )(x)

        x_out = self.act_fn(z + x)
        return x_out


class PreActResNetBlock(ResNetBlock):

    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    strides=(1, 1) if not self.subsample else (2, 2),
                    kernel_init=conv_kernel_init,
                    use_bias=False)(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(self.c_out, kernel_size=(3, 3),
                    kernel_init=conv_kernel_init,
                    use_bias=False)(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(self.c_out,
                        kernel_size=(1, 1),
                        strides=(2, 2),
                        kernel_init=conv_kernel_init,
                        use_bias=False)(x)

        x_out = z + x
        return x_out


class ResNetMod(nn.Module):
    num_classes : int
    act_fn : callable
    block_class : nn.Module
    num_blocks : tuple = (3, 3, 3)
    c_hidden : tuple = (16, 32, 64)

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        # A first convolution on the original image to scale up the channel size
        x = nn.Conv(
            self.c_hidden[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            kernel_init=conv_kernel_init,
            use_bias=False
        )(x)
        # x = nn.Conv(self.c_hidden[0], kernel_size=(3, 3), kernel_init=conv_kernel_init, use_bias=False)(x)  # flax default
        if self.block_class == ResNetBlock:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)

        # Creating the ResNet blocks
        for block_idx, block_count in enumerate(self.num_blocks):
            for bc in range(block_count):
                # Subsample the first block of each group, except the very first one.
                subsample = (bc == 0 and block_idx > 0)
                # ResNet block
                x = self.block_class(
                    c_out=self.c_hidden[block_idx],
                    act_fn=self.act_fn,
                    subsample=subsample
                    )(x, train=train)

        # Mapping to classification output
        feature_map = x.mean(axis=(1, 2))

        x = nn.Dense(self.num_classes)(feature_map)

        if feature:
            return (x, feature_map)

        return x


URLS = {'resnet18': 'https://www.dropbox.com/s/wx3vt76s5gpdcw5/resnet18_weights.h5?dl=1',
        'resnet34': 'https://www.dropbox.com/s/rnqn2x6trnztg4c/resnet34_weights.h5?dl=1',
        'resnet50': 'https://www.dropbox.com/s/fcc8iii38ezvqog/resnet50_weights.h5?dl=1',
        'resnet101': 'https://www.dropbox.com/s/hgtnk586pnz0xug/resnet101_weights.h5?dl=1',
        'resnet152': 'https://www.dropbox.com/s/tvi28uwiy54mcfr/resnet152_weights.h5?dl=1'}

LAYERS = {'resnet18': [2, 2, 2, 2],
          'resnet34': [3, 4, 6, 3],
          'resnet50': [3, 4, 6, 3],
          'resnet101': [3, 4, 23, 3],
          'resnet152': [3, 8, 36, 3]}


class BasicBlock(nn.Module):
    """
    Basic Block.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    downsample: bool=False
    stride: bool=True
    param_dict: h5py.Group=None
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    block_name: str=None
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Basic Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x 
        
        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size, 
                    strides=(2, 2) if self.downsample else (1, 1),
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features, 
                    kernel_size=self.kernel_size, 
                    strides=(1, 1), 
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv2']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn2'],
                           dtype=self.dtype) 

        if self.downsample:
            residual = nn.Conv(features=self.features, 
                               kernel_size=(1, 1), 
                               strides=(2, 2), 
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['downsample']['conv']['weight']), 
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = ops.batch_norm(residual,
                                      train=train,
                                      epsilon=1e-05,
                                      momentum=0.1,
                                      params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                                      dtype=self.dtype) 
        
        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class Bottleneck(nn.Module):
    """
    Bottleneck.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        expansion (int): Factor to multiply number of output channels with.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]]=(3, 3)
    downsample: bool=False
    stride: bool=True
    param_dict: Any=None
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    block_name: str=None
    expansion: int=4
    dtype: str='float32'

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Bottleneck.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x 
        
        x = nn.Conv(features=self.features, 
                    kernel_size=(1, 1), 
                    strides=(1, 1),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features, 
                    kernel_size=(3, 3), 
                    strides=(2, 2) if self.downsample and self.stride else (1, 1), 
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv2']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)
        
        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn2'],
                           dtype=self.dtype) 
        x = nn.relu(x)

        x = nn.Conv(features=self.features * self.expansion, 
                    kernel_size=(1, 1), 
                    strides=(1, 1), 
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv3']['weight']), 
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn3'],
                           dtype=self.dtype) 

        if self.downsample:
            residual = nn.Conv(features=self.features * self.expansion, 
                               kernel_size=(1, 1), 
                               strides=(2, 2) if self.stride else (1, 1), 
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['downsample']['conv']['weight']), 
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = ops.batch_norm(residual,
                                      train=train,
                                      epsilon=1e-05,
                                      momentum=0.1,
                                      params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                                      dtype=self.dtype) 
        
        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class ResNet(nn.Module):
    """
    ResNet.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        architecture (str): 
            Which ResNet model to use:
                - 'resnet18'
                - 'resnet34'
                - 'resnet50'
                - 'resnet101'
                - 'resnet152'
        num_classes (int):
            Number of classes.
        block (nn.Module):
            Type of residual block:
                - BasicBlock
                - Bottleneck
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.
    """
    output: str='softmax'
    pretrained: str='imagenet'
    normalize: bool=True
    architecture: str='resnet18'
    num_classes: int=1000
    block: nn.Module=BasicBlock
    kernel_init: functools.partial=nn.initializers.lecun_normal()
    bias_init: functools.partial=nn.initializers.zeros
    ckpt_dir: str=None
    dtype: str='float32'

    def setup(self):
        self.param_dict = None
        if self.pretrained == 'imagenet':
            ckpt_file = utils.download(self.ckpt_dir, URLS[self.architecture])
            self.param_dict = h5py.File(ckpt_file, 'r')

    @nn.compact
    def __call__(self, x, train=True, feature=False):
        """
        Args:
            x (tensor): Input tensor of shape [N, H, W, 3]. Images must be in range [0, 1].
            train (bool): Training mode.

        Returns:
            (tensor): Out
            If output == 'logits' or output == 'softmax':
                (tensor): Output tensor of shape [N, num_classes].
            If output == 'activations':
                (dict): Dictionary of activations.
        """
        if self.normalize:
            mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1).astype(self.dtype)  # EDITED
            std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1).astype(self.dtype)  # EDITED
            x = (x - mean) / std

        if self.pretrained == 'imagenet':
            # if self.num_classes != 1000: # EDITED
            #     warnings.warn(f'The user specified parameter \'num_classes\' was set to {self.num_classes} ' # EDITED
            #                     'but will be overwritten with 1000 to match the specified pretrained checkpoint \'imagenet\', if ', UserWarning) # EDITED
            # num_classes = 1000 # EDITED
            num_classes = self.num_classes # EDITED
        else:
            num_classes = self.num_classes
 
        act = {}

        x = nn.Conv(features=64, 
                    kernel_size=(7, 7),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_ : jnp.array(self.param_dict['conv1']['weight']),
                    strides=(2, 2), 
                    padding=((3, 3), (3, 3)),
                    use_bias=False,
                    dtype=self.dtype)(x)
        act['conv1'] = x

        x = ops.batch_norm(x,
                           train=train,
                           epsilon=1e-05,
                           momentum=0.1,
                           params=None if self.param_dict is None else self.param_dict['bn1'],
                           dtype=self.dtype)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Layer 1
        down = self.block.__name__ == 'Bottleneck'
        for i in range(LAYERS[self.architecture][0]):
            params = None if self.param_dict is None else self.param_dict['layer1'][f'block{i}']
            x = self.block(features=64,
                           kernel_size=(3, 3),
                           downsample=i == 0 and down,
                           stride=i != 0,
                           param_dict=params,
                           block_name=f'block1_{i}',
                           dtype=self.dtype)(x, act, train)
        
        # Layer 2
        for i in range(LAYERS[self.architecture][1]):
            params = None if self.param_dict is None else self.param_dict['layer2'][f'block{i}']
            x = self.block(features=128,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block2_{i}',
                           dtype=self.dtype)(x, act, train)
        
        # Layer 3
        for i in range(LAYERS[self.architecture][2]):
            params = None if self.param_dict is None else self.param_dict['layer3'][f'block{i}']
            x = self.block(features=256,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block3_{i}',
                           dtype=self.dtype)(x, act, train)

        # Layer 4
        for i in range(LAYERS[self.architecture][3]):
            params = None if self.param_dict is None else self.param_dict['layer4'][f'block{i}']
            x = self.block(features=512,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block4_{i}',
                           dtype=self.dtype)(x, act, train)

        # Classifier
        x = jnp.mean(x, axis=(1, 2))

        feature_map = x

        x = nn.Dense(features=num_classes,
                     kernel_init=self.kernel_init if (self.param_dict is None or self.num_classes != 1000) else lambda *_ : jnp.array(self.param_dict['fc']['weight']),  # EDITED
                     bias_init=self.bias_init if (self.param_dict is None or self.num_classes != 1000)  else lambda *_ : jnp.array(self.param_dict['fc']['bias']),  # EDITED
                     dtype=self.dtype)(x)
        act['fc'] = x
        
        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'activations':
            return act

        if feature:
            return (x, feature_map)
        else:
            return x


def ResNet18(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet18 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet18',
                  num_classes=num_classes,
                  block=BasicBlock,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet34(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet34 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet34',
                  num_classes=num_classes,
                  block=BasicBlock,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet50(output='softmax',
             pretrained='imagenet',
             normalize=True,
             num_classes=1000,
             kernel_init=nn.initializers.lecun_normal(),
             bias_init=nn.initializers.zeros,
             ckpt_dir=None,
             dtype='float32'):
    """
    Implementation of the ResNet50 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet50',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet101(output='softmax',
              pretrained='imagenet',
              normalize=True,
              num_classes=1000,
              kernel_init=nn.initializers.lecun_normal(),
              bias_init=nn.initializers.zeros,
              ckpt_dir=None,
              dtype='float32'):
    """
    Implementation of the ResNet101 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet101',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


def ResNet152(output='softmax',
              pretrained='imagenet',
              normalize=True,
              num_classes=1000,
              kernel_init=nn.initializers.lecun_normal(),
              bias_init=nn.initializers.zeros,
              ckpt_dir=None,
              dtype='float32'):
    """
    Implementation of the ResNet152 by He et al.
    Reference: https://arxiv.org/abs/1512.03385

    The pretrained parameters are taken from:
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    
    Args:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000] 
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        num_classes (int):
            Number of classes.
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used. 
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.

    Returns:
        (nn.Module): ResNet network.
    """
    return ResNet(output=output,
                  pretrained=pretrained,
                  normalize=normalize,
                  architecture='resnet152',
                  num_classes=num_classes,
                  block=Bottleneck,
                  kernel_init=kernel_init,
                  bias_init=bias_init,
                  ckpt_dir=ckpt_dir,
                  dtype=dtype)


rng_key = main_rng

if 'CNN' in model_name:
    model_class = CNN
    num_blocks = None
    c_hidden = None
if 'MLP_Toy' in model_name:
    model_class = MLP_Toy
    num_blocks = None
    c_hidden = None
if 'ResNet9' in model_name:  # 272,896 parameters for FMNIST
    model_class = ResNetMod
    num_blocks = (3, 3, 3)
    c_hidden = (16, 32, 64)
if 'ResNet18' in model_name:  # 11,174,642 parameters for FMNIST
    model_class = ResNetMod
    num_blocks = (2, 2, 2, 2)
    c_hidden = (64, 128, 256, 512)
if 'ResNet50' in model_name:
    model_class = ResNetMod
    num_blocks = None
    c_hidden = (64, 128, 256, 512)

block_class = ResNetBlock
# block_class = PreActResNetBlock

act_fn = nn.relu
# act_fn = nn.swish

if prior_precision == 0:
    prior_precision = 1 / prior_var
elif prior_var == 0:
    prior_var = 1 / prior_precision
else:
    raise ValueError("Only one of prior_precision and prior_var can be set.")

prior_mean = "Pretrained Mean" if "Pretrained" in model_name else prior_mean

if method == "fsmap":
    stochastic = False
if method == "psmap":
    stochastic = False
if method == "psvi":
    stochastic = True

resnet_trainer, resnet_results = trainer(
    model_name=model_name,
    model_class=model_class,
    model_hparams={
                        "num_classes": num_classes,
                        "c_hidden": c_hidden,
                        "num_blocks": num_blocks,
                        "act_fn": act_fn,
                        "block_class": block_class,
                        },
    optimizer_name=optimizer_name,
    optimizer_hparams={
                        "learning_rate": learning_rate,
                        "learning_rate_scale_logvar": learning_rate_scale_logvar,
                        "momentum": momentum,
                        "alpha": alpha,
                        "weight_decay": weight_decay,
                        },
    objective_hparams={
                        "method": method,
                        "stochastic": stochastic,
                        "reg_type": reg_type,
                        "reg_scale": reg_scale,
                        "prior_mean": prior_mean,
                        "prior_var": prior_var,
                        "prior_likelihood_scale": prior_likelihood_scale,
                        "prior_likelihood_f_scale": prior_likelihood_f_scale,
                        "prior_likelihood_cov_scale": prior_likelihood_cov_scale,
                        "prior_likelihood_cov_diag": prior_likelihood_cov_diag,
                        "likelihood_scale": likelihood_scale,
                        "context_points": context_points,
                        "forward_points": forward_points,
                        "reg_points": reg_points,
                        "mc_samples_llk": mc_samples_llk,
                        "mc_samples_reg": mc_samples_reg,
                        "training_dataset_size": training_dataset_size,
                        "batch_size": batch_size,
                        "init_logvar": init_logvar,
                        "init_final_layer_weights_logvar": init_final_layer_weights_logvar,
                        "init_final_layer_bias_logvar": init_final_layer_bias_logvar,
                        "prior_feature_logvar": prior_feature_logvar,
                        "pretrained_prior": pretrained_prior,
                        },
    other_hparams={
                        "output_var": output_var,
                        "stochastic": stochastic,
                        "evaluate": evaluate,
                        "restore_checkpoint": restore_checkpoint,
                        "batch_stats_init_epochs": batch_stats_init_epochs,
                        "dataset": dataset,
                        "prediction_type": prediction_type,
                        "ood_points": ood_points,
                        "batch_size": batch_size,
                        "context_batch_size": context_batch_size,
                        "context_dataset_size": context_dataset_size,
                        "num_epochs": num_epochs,
                        "seed": seed,
                        "mc_samples_eval": mc_samples_eval,
                        "config_name": config_name,
                        "debug_print": debug_print,
                        "log_frequency": log_frequency,
                        "full_eval": full_eval,
                        "save_to_wandb": save_to_wandb,
                        "wandb_project": wandb_project,
                        "wandb_account": wandb_account,
                        },
    exmp_inputs=jax.device_put(
        next(iter(train_loader))[0]),
    rng_key=rng_key,
    )


print(f"\nDone\n")
