from datetime import datetime
import itertools
from joblib import Parallel, delayed
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from iforest import BAD_IForest
import odds_datasets


def run_sim(
    dataset,
    batch_size,
    strategy,
    data_copies,
    seed,
):
    # load dataset
    X, y = odds_datasets.load(dataset)
    contamination = y.mean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )
    X_train = np.concatenate([X_train] * data_copies, axis=0)
    y_train = np.concatenate([y_train] * data_copies, axis=0)

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
    save_dir = f"results/{dataset}x{data_copies}/{strategy}/batch_size_{batch_size}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/seed_{seed}_{current_time}"
    np.savez_compressed(save_path, avp_train=np.array(avp_train), avp_test=np.array(avp_test))


if __name__ == "__main__":
    # run small datasets
    Parallel(n_jobs=10)(
        delayed(run_sim)(
            dataset=dataset,
            batch_size=batch_size,
            strategy=strategy,
            seed=seed,
            data_copies=copies,
        )
        for dataset in odds_datasets.small_datasets_names + odds_datasets.medium_datasets_names
        for copies in [1, 2, 5, 10]
        for batch_size in [1, 2, 5, 10]
        for strategy in ["worstcase", "average", "bestcase", "independent"]
        for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    # run large datasets
    Parallel(n_jobs=10)(
        delayed(run_sim)(
            dataset=dataset,
            batch_size=batch_size,
            strategy=strategy,
            seed=seed,
            data_copies=copies,
        )
        for dataset in odds_datasets.small_datasets_names + odds_datasets.medium_datasets_names
        for copies in [1, 2]
        for batch_size in [1, 2, 5]
        for strategy in ["worstcase", "average", "independent"]
        for seed in [0, 1, 2, 3, 4]
    )
