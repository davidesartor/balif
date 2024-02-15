import scipy.io
import os
import jax
import jax.numpy as jnp

from sklearn.model_selection import train_test_split

small_datasets_names = [
    "wine",
    "vertebral",
    "ionosphere",
    "wbc",
    "breastw",
    "pima",
]

medium_datasets_names = ["vowels", "letter", "cardio", "thyroid"]

large_datasets_names = [
    "optdigits",
    "satimage-2",
    "satellite",
    "pendigits",
    "annthyroid",
    "mnist",
    "mammography",
    "cover",
]

datasets_names = small_datasets_names + medium_datasets_names + large_datasets_names


def load(dataset_name=None):
    mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), f"{dataset_name}.mat"))
    data, labels = mat["X"], mat["y"][:, 0]
    return jnp.array(data, dtype=float), jnp.array(labels, dtype=bool)


def load_as_train_test(dataset_name=None, **kwargs):
    data, labels = load(dataset_name)
    return train_test_split(data, labels, stratify=labels, **kwargs)
