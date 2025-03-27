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
    scores = [model.decision_function(X)]
    avp = [average_precision_score(y, scores[-1])]
    queries = []

    # run the simulation
    iterations = int(np.ceil(X.shape[0] / batch_size))
    queriable = np.ones(X.shape[0], dtype=bool)
    for _ in tqdm(range(iterations), f"{dataset}: strat={strategy} bs={batch_size}"):
        idxs = model.get_queries(X[queriable], batch_size)
        queriable[queriable][idxs] = False
        model.update(X[idxs, :], y[idxs])

        queries.append(idxs)
        scores.append(model.decision_function(X))
        avp.append(average_precision_score(y, scores[-1]))

    # save results
    save_dir = f"results/{dataset}/{strategy}/batch_size_{batch_size}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/seed_{seed}_{current_time}_avp.txt"
    np.savez_compressed(
        save_path,
        avp=np.array(avp),
        scores=np.array(scores),
        queries=np.array(queries),
    )


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    batch_sizes = [1, 2, 5, 10]
    strategies = ["worstcase", "average"]

    Parallel(n_jobs=8)(
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
