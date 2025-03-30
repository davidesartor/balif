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
    # load dataset
    (X_train, X_test, y_train, y_test) = odds_datasets.load_as_train_test(
        dataset, random_state=seed, test_size=0.5
    )
    contamination = (y_train.mean() + y_test.mean()) / 2

    # fit the unsupervised model
    model = BAD_IForest(
        contamination=contamination,
        batch_query_method=strategy,
        random_state=seed,
    ).fit(X_train)

    # get and save unsupervised average precision
    scores_train = model.decision_function(X_train)
    scores_test = model.decision_function(X_test)
    avp_train = [average_precision_score(y_train, scores_train)]
    avp_test = [average_precision_score(y_test, scores_test)]

    # run the simulation
    iterations = int(np.floor(len(X_train) / batch_size))
    queriable = np.ones(len(X_train), dtype=bool)
    for _ in tqdm(range(iterations), f"{dataset}: strat={strategy} bs={batch_size}"):
        idxs = model.get_queries(X_train[queriable], batch_size)
        idxs = np.arange(len(X_train))[queriable][idxs]
        queriable[idxs] = False
        model.update(X_train[idxs, :], y_train[idxs])

        scores_train = model.decision_function(X_train)
        scores_test = model.decision_function(X_test)
        avp_train.append(average_precision_score(y_train, scores_train))
        avp_test.append(average_precision_score(y_test, scores_test))

    # query the remaining samples
    if np.any(queriable):
        idxs = np.arange(len(X_train))[queriable]
        model.update(X_train[idxs], y_train[idxs])
        scores_train = model.decision_function(X_train)
        scores_test = model.decision_function(X_test)
        avp_train.append(average_precision_score(y_train, scores_train))
        avp_test.append(average_precision_score(y_test, scores_test))

    # save results
    save_dir = f"results/{dataset}/{strategy}/batch_size_{batch_size}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/seed_{seed}_{current_time}"
    np.savez_compressed(
        save_path,
        avp_train=np.array(avp_train),
        avp_test=np.array(avp_test)
    )


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_sizes = [1, 2, 5, 10]
    strategies = ["worstcase", "average", "independent"]

    Parallel(n_jobs=10)(
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
