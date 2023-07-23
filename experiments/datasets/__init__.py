import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

from sklearn.model_selection import train_test_split

small_datasets_names = [
    "wine",
    "vertebral",
    "ionosphere",
    "wbc",
    "squarethoroid",
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


def load(dataset_name=None, dtype=np.float64, **kwargs):
    if dataset_name is None or dataset_name == "squarethoroid":

        def make_square_toroid(k=1, cyrcle=False, random_state=None, **kwargs):
            if random_state is not None:
                np.random.seed(random_state)
            if cyrcle:
                cluster = np.random.uniform(-0.8, 0.8, [int(1000 * k), 2])
                central_cluster = cluster[
                    np.abs(np.sum(np.square(cluster), axis=1) - 0.45) < 0.2
                ]
                anomaly = cluster[np.sum(np.square(cluster), axis=1) < 0.2][::3]
            else:
                central_cluster = np.random.uniform(-0.8, 0.8, [int(1000 * k), 2])
                central_cluster = central_cluster[
                    np.any(np.abs(central_cluster) > 0.6, axis=1)
                ]
                anomaly = np.random.uniform(-0.55, 0.55, [int(50 * k), 2])

            data = np.vstack([central_cluster, anomaly])
            labels = np.hstack(
                [np.zeros(central_cluster.shape[0]), np.ones(anomaly.shape[0])]
            )

            return data, labels

        data, labels = make_square_toroid(**kwargs)
    else:
        mat = scipy.io.loadmat(
            os.path.join(os.path.dirname(__file__), f"{dataset_name}.mat")
        )
        data, labels = mat["X"], mat["y"][:, 0]
    return data.astype(dtype), labels.astype(dtype)


def load_as_train_test(dataset_name=None, dtype=np.float64, **kwargs):
    data, labels = load(dataset_name, dtype, **kwargs)
    return train_test_split(data, labels, stratify=labels, **kwargs)
