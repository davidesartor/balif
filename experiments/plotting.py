from numpy.typing import NDArray
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from matplotlib import pyplot as plt

@runtime_checkable
class Model(Protocol):
    def fit(self, data: NDArray[np.float64], seed: Optional[int] = None) -> None:
        ...

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        ...


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
