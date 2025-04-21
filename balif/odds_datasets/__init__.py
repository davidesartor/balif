import scipy.io
import os
import numpy as np

small_datasets_names = [
    "wine",
    "vertebral",
    "ionosphere",
    "wbc",
    "breastw",
    "pima",
]

medium_datasets_names = [
    "vowels",
    "letter",
    "cardio",
    "thyroid",
]

large_datasets_names = [
    "optdigits",
    "satimage-2",
    "satellite",
    "pendigits",
    "annthyroid",
    "mnist",
    "mammography",
    # "cover",
]

datasets_names = small_datasets_names + medium_datasets_names + large_datasets_names


def load(dataset_name=None, scale=False):
    mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), f"{dataset_name}.mat"))
    data, labels = mat["X"], mat["y"][:, 0]
    if scale:
        data = (data - data.mean(axis=0)) / data.std(axis=0)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=bool)
