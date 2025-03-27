from datetime import datetime
import itertools
from joblib import Parallel, delayed
import numpy as np
import os
import random
from tqdm import tqdm


from sklearn.metrics import average_precision_score
from iforest import BAD_IForest
import odds_datasets


def run_sim(
    dataset,
    batch_size,
    strategy,
    seed,
):
    # create save path
    save_dir = f"results/{dataset}/{strategy}/batch_size_{batch_size}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/seed_{seed}_{current_time}_avp.txt"

    # set seeds
    np.random.seed(seed)
    random.seed(seed)

    # load dataset
    X, y = odds_datasets.load(dataset)
    contamination = y.mean()
    
    # fit the unsupervised model
    model = BAD_IForest(
        contamination=contamination,
        batch_query_method=strategy,
    ).fit(X)

    # get and save unsupervised average precision
    scores0 = model.decision_function(X)
    avp_0 = average_precision_score(y, scores0)
    with open(save_path, "a") as f:
        f.write(f"{avp_0}\n")

    # run the simulation
    iterations = int(np.ceil(X.shape[0] / batch_size))
    queriable = np.ones(X.shape[0], dtype=bool)
    for _ in tqdm(range(iterations), f"{dataset}: strat={strategy} bs={batch_size}"):
        batch_idxs = model.get_queries(X[queriable], batch_size)
        queriable[batch_idxs] = False
        model.update(X[batch_idxs, :], y[batch_idxs])
        scores = model.decision_function(X)
        avp = average_precision_score(y, scores)
        with open(save_path, "a") as f:
            f.write(f"{avp}\n")


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    batch_sizes = [1, 2, 5, 10]
    strategies = ["worstcase", "average"]

    Parallel(n_jobs=32)(
        delayed(run_sim)(
            dataset=dataset,
            batch_size=batch_size,
            strategy=strategy,
            seed=seed,
        )
        for dataset, batch_size, strategy, seed in itertools.product(
            odds_datasets.datasets_names, batch_sizes, strategies, seeds
        )
    )
