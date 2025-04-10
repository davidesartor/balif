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
    data_copies,
    batch_size,
    interest_method,
    independent_queries,
    seed,
):
    # load dataset
    X, y = odds_datasets.load(dataset)
    contamination = y.mean()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    # duplicate trainset
    X_train = np.concatenate([X_train] * data_copies, axis=0)
    y_train = np.concatenate([y_train] * data_copies, axis=0)

    # fit the unsupervised model
    model = BAD_IForest(
        interest_method=interest_method,
        contamination=contamination,
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
    for _ in tqdm(range(iterations), f"{dataset}: strat={interest_method} bs={batch_size}"):
        # get the queries indices
        idxs = model.get_queries(
            X=X_train,
            batch_size=batch_size,
            independent=independent_queries,
            mask=queriable,
        )

        # update the model with the queried samples
        model.update(X_train[idxs, :], y_train[idxs])
        queriable[idxs] = False

        # get the average precision scores
        scores_train = model.decision_function(X_train)
        scores_test = model.decision_function(X_test)
        avp_train.append(average_precision_score(y_train, scores_train))
        avp_test.append(average_precision_score(y_test, scores_test))

    if np.any(queriable):
        # query the remaining samples
        idxs = np.arange(len(X_train))[queriable]
        model.update(X_train[idxs, :], y_train[idxs])
        queriable[idxs] = False

        # get the average precision scores
        scores_train = model.decision_function(X_train)
        scores_test = model.decision_function(X_test)
        avp_train.append(average_precision_score(y_train, scores_train))
        avp_test.append(average_precision_score(y_test, scores_test))

    # save results
    assert not queriable.any(), "Some points have not been queried"
    save_dir = f"results/{data_copies}x{dataset}/{"independent" if independent_queries else "worstcase"}/{interest_method}/batch_size_{batch_size}"
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass
    save_path = f"{save_dir}/seed_{seed}"
    np.savez_compressed(save_path, avp_train=np.array(avp_train), avp_test=np.array(avp_test))


if __name__ == "__main__":
    # run small datasets
    Parallel(n_jobs=16)(
        delayed(run_sim)(
            dataset=dataset,
            data_copies=batch_size,
            batch_size=batch_size,
            interest_method=interest,
            independent_queries=independent,
            seed=seed,
        )
        for dataset in odds_datasets.small_datasets_names + odds_datasets.medium_datasets_names
        for interest in ["bald", "margin", "anom"]
        for independent in [True, False]
        for batch_size in [1, 5, 10]
        for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
