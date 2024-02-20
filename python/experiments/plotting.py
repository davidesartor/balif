import os
import pickle
import zlib
from numpy.typing import NDArray
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve

from .utils import extract_ap_matrix, extract_fbeta_matrix, extract_label_matrix, extract_scores_matrix


@runtime_checkable
class Model(Protocol):
    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None) -> None:
        ...

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        ...


def confidence_interval_95(
    data: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ci = 1.96 * std / np.sqrt(len(data))
    return mean, mean - ci, mean + ci


def heatmap_2d(
    model: Model,
    range_x: tuple[int, int] = (-1, 1),
    range_y: tuple[int, int] = (-1, 1),
    range_score: Optional[tuple[int, int]] = None,
    marked_outliers: Optional[NDArray[np.float64]] = None,
    marked_inliers: Optional[NDArray[np.float64]] = None,
    n_grid_points: int = 50,
    is_stand_alone_figure: bool = True,
    title: Optional[str] = None,
):
    if is_stand_alone_figure:
        plt.figure()
    if title is not None:
        plt.title(title)

    grid = np.meshgrid(
        np.linspace(*range_x, n_grid_points),
        np.linspace(*range_y, n_grid_points),
    )
    grid_scores = model.predict(np.array(grid).reshape(2, -1).T)
    plt.imshow(
        np.array(grid_scores).reshape(n_grid_points, n_grid_points),
        cmap="coolwarm",
        vmin=range_score[0] if range_score is not None else min(grid_scores),
        vmax=range_score[1] if range_score is not None else max(grid_scores),
        extent=(*range_x, *range_y),
        origin="lower",
    )
    plt.colorbar()
    if marked_outliers is not None:
        plt.scatter(
            *marked_outliers.T,
            facecolors="none",
            edgecolors="firebrick",
            label="labelled outliers"
        )
    if marked_inliers is not None:
        plt.scatter(
            *marked_inliers.T,
            facecolors="none",
            edgecolors="navy",
            label="labelled inliers"
        )

    if is_stand_alone_figure:
        plt.show()


def plot_ap_evol(dataset_name: str, model_name: str) -> None:
    with open(f"sim_results/{model_name}/{dataset_name}.gz", 'rb') as fp:
        data = zlib.decompress(fp.read())
        sim_res = pickle.loads(data)
        ap_avg, ap_low_ci, ap_high_ci = confidence_interval_95(extract_ap_matrix(sim_res))
        plt.plot(ap_avg, label=model_name)
        plt.fill_between(np.arange(len(ap_avg)), ap_low_ci, ap_high_ci, alpha=0.2)  # type: ignore
        plt.ylim(0, 1.01)
        plt.xlim(0, len(ap_avg)-1)
    plt.xlabel("Number of labelled datapoints")
    plt.ylabel("Average precision")
    

def plot_fbeta_evol(dataset_name: str, model_name: str, beta: float = 1.0) -> None:
    with open(f"sim_results/{model_name}/{dataset_name}.gz", 'rb') as fp:
        data = zlib.decompress(fp.read())
        sim_res = pickle.loads(data)
        f1_avg, f1_low_ci, f1_high_ci = confidence_interval_95(extract_fbeta_matrix(sim_res, beta))
        plt.plot(f1_avg, label=model_name)
        plt.fill_between(np.arange(len(f1_avg)), f1_low_ci, f1_high_ci, alpha=0.2)  # type: ignore
        plt.ylim(0, 1.01)
        plt.xlim(0, len(f1_avg)-1)
        plt.xlabel("Number of labelled datapoints")
        plt.ylabel(f"F{beta} score")
        
def plot_prec_recall_evolution(dataset_name: str, model_name: str, run: int | None = None) -> None:
    with open(f"sim_results/{model_name}/{dataset_name}.gz", 'rb') as fp:
        data = zlib.decompress(fp.read())
        sim_results = pickle.loads(data)

        score_matrix = extract_scores_matrix(sim_results)
        label_matrix = extract_label_matrix(sim_results)
        if run is None:
            run = np.random.randint(len(score_matrix))

        score_matrix = score_matrix[run]
        labels = label_matrix[run]

        def max_fbeta(labels, scores, beta=1.0):
            precision, recall, _ = precision_recall_curve(labels, scores)
            fbeta = np.divide((1+beta**2)*recall*precision,
                              recall+beta**2*precision+1e-8)
            return max(fbeta), recall[np.argmax(fbeta)], precision[np.argmax(fbeta)]

        for iter, scores in enumerate(score_matrix):
            if not (iter == 0 or iter % 5 == 0):
                continue
            precision, recall, _ = precision_recall_curve(labels, scores)
            max_f1, optimal_recall, optimal_precision = max_fbeta(
                labels, scores)
            plt.plot(recall, precision, color=plt.cm.viridis(iter / (len(score_matrix)-1)), label = f"{iter} queries, F1={max_f1:.2f}" if iter == 0 or iter == len(score_matrix)-1 else None) #type: ignore
            # plot hollow circle at optimal point
            plt.plot(optimal_recall, optimal_precision, "o", color=plt.cm.viridis(iter / (len(score_matrix)-1))) #type: ignore
            plt.ylim(0, 1.01)
            plt.xlim(0, 1.01)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        
