from numpy.typing import NDArray
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from sklearn.metrics import average_precision_score

from .plotting import heatmap_2d


@runtime_checkable
class UpdatableModel(Protocol):
    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None) -> None:
        ...

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def update(
        self,
        *args,
        inlier_data: Optional[NDArray[np.float64]] = None,
        outlier_data: Optional[NDArray[np.float64]] = None,
    ) -> None:
        ...


def run_sim(
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

    sim_results = {"true_labels": labels_test}
    is_queried = np.zeros(data_train.shape[0], dtype=bool)
    for iter in range(n_iter):
        if iter == 0:
            model.fit(data_train, seed=seed)
        else:
            query_priority = np.argsort(model.predict(data_train))[::-1]
            #first index of query_priority that is not yet queried
            query_idx = 0 # reduntant, but avoids needing multiple type ignore for mypy
            for query_idx in query_priority:
                if is_queried[query_idx] == False:
                    break
            if labels_train[query_idx] == 1:
                model.update(outlier_data=data_train[query_idx]) 
            else:
                model.update(inlier_data=data_train[query_idx]) 
            is_queried[query_idx] = True

        if show_heatmap_evol and iter % 5 == 0:
            queried_data = data_train[is_queried == True]
            queried_labels = labels_train[is_queried == True]
            heatmap_2d(
                model,
                title=f"iter {iter}",
                marked_outliers=queried_data[queried_labels == 1],
                marked_inliers=queried_data[queried_labels == 0],
            )

        scores_train = model.predict(data_train)
        if data_test is not None:
            scores_test = model.predict(data_test)
        else:
            scores_test = scores_train
    
        average_precision_test = average_precision_score(labels_test, scores_test)

        sim_results[f"scores_{iter}_queries"] = scores_test
        sim_results[f"AP_{iter}_queries"] = np.array(average_precision_test)

    return sim_results
