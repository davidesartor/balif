from numpy.typing import NDArray
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from sklearn.metrics import average_precision_score
from tqdm.notebook import tqdm
import joblib

from .datasets import load_as_train_test
from .plotting import heatmap_2d


@runtime_checkable
class UpdatableModel(Protocol):
    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None) -> None:
        ...

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def update(
        self,
        *,
        inlier_data: Optional[NDArray[np.float64]] = None,
        outlier_data: Optional[NDArray[np.float64]] = None,
    ) -> None:
        ...


def simulate_run(
    model: UpdatableModel,
    n_iter: int,
    data_train: NDArray[np.float64],
    labels_train: NDArray[np.float64],
    data_test: Optional[NDArray[np.float64]] = None,
    labels_test: Optional[NDArray[np.float64]] = None,
    seed: Optional[int] = None,
    show_heatmap_evol: bool = False,
):
    """fit model using only train_data and train_labels, then iteratively update model.
    queries use the "most anomalous" strategy and only consider points in training_data.
    use test data to evaluate model performance."""
    if labels_test is None:
        labels_test = labels_train
    if data_test is None:
        data_test = data_train

    sim_results = {
        "n_iter": n_iter,
        "train": {"data": data_train, "labels": labels_train},
        "test": {"data": data_test, "labels": labels_test},
    }

    is_queried = np.zeros(data_train.shape[0], dtype=bool)
    for iter in range(n_iter + 1):
        if iter == 0:
            model.fit(data_train, seed=seed)
        else:
            query_priority = np.argsort(model.predict(data_train))[::-1]
            # first index of query_priority that is not yet queried
            query_idx = 0  # reduntant, but avoids needing multiple type ignore for mypy
            for query_idx in query_priority:
                if is_queried[query_idx] == False:
                    break
            if labels_train[query_idx] == 1:
                model.update(outlier_data=data_train[query_idx])
            else:
                model.update(inlier_data=data_train[query_idx])
            is_queried[query_idx] = True

        queried_data = data_train[is_queried == True]
        queried_labels = labels_train[is_queried == True]

        if show_heatmap_evol and iter % 5 == 0:
            heatmap_2d(
                model,
                title=f"iter {iter}",
                marked_outliers=queried_data[queried_labels == 1],
                marked_inliers=queried_data[queried_labels == 0],
            )

        scores_train = model.predict(data_train)
        scores_test = model.predict(data_test)
        ap_train = average_precision_score(labels_train, scores_train)
        ap_test = average_precision_score(labels_test, scores_test)

        sim_results["train"][f"after_{iter}_queries"] = {
            "scores": scores_train,
            "ap": ap_train,
            "queried_data": queried_data,
            "queried_labels": queried_labels,
        }
        sim_results["test"][f"after_{iter}_queries"] = {
            "scores": scores_test,
            "ap": ap_test,
        }

    return sim_results


def multi_run_simulation(
    model: UpdatableModel,
    n_runs: int,
    n_iter: int,
    dataset_name: str,
    test_size=0.33,
    parallel_jobs: int = 1,
):
    def run_setup(run_idx: int):
        data_train, data_test, labels_train, labels_test = load_as_train_test(
            dataset_name=dataset_name, random_state=run_idx, test_size=test_size
        )
        return f"run_{run_idx}", simulate_run(
            model=model,
            n_iter=n_iter,
            data_train=data_train,
            labels_train=labels_train,
            data_test=data_test,
            labels_test=labels_test,
            seed=run_idx,
        )

    if parallel_jobs == 1:
        run_results = [run_setup(run_idx) for run_idx in tqdm(range(n_runs), leave=False)]
    else:
        run_results = joblib.Parallel(n_jobs=parallel_jobs)(
            joblib.delayed(run_setup)(run_idx) for run_idx in tqdm(range(n_runs), leave=False)
        )
    return {key: res for key, res in run_results}  # type: ignore
