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

save_dir = "batch_results/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def run_sim(X, y, batch_size, strategy, seed=0, contamination_factor=0.1, query_multiple=False, dataset_name=None): 

    # set save file name
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{save_dir}/{dataset_name}_bs_{batch_size}_{strategy}_seed_{seed}_{current_time}_avp.txt"

    # set seeds
    np.random.seed(seed)
    random.seed(seed) 

    # fit the unsupervised model
    model = BAD_IForest(contamination=contamination_factor).fit(X)
    
    # get and save unsupervised average precision
    scores0 = model.decision_function(X)
    avp_0 = average_precision_score(y, scores0)
    with open(save_path, "a") as f:
        f.write(f"{avp_0}\n")

    iterations = int(np.ceil(X.shape[0] / batch_size))
    queriable = np.ones(X.shape[0], dtype=bool)

    for _ in range(iterations): 
        batch_idxs = model.get_queries(X[queriable], batch_size)
        queriable[batch_idxs] = False
        model.update(X[batch_idxs,:], y[batch_idxs])
        scores = model.decision_function(X)
        avp = average_precision_score(y, scores)
        with open(save_path, "a") as f:
            f.write(f"{avp}\n")
        
    

def main(): 
    seeds = [0, 1] # 2, 3, 4]
    datasets = ['wine', 'pima', 'cardio', 'annthyroid']
    batch_sizes = [1, 2, 5, 10 ]
    strategies = ['wc', 'avg']

    
    configs = list(itertools.product(seeds, datasets, batch_sizes, strategies))

    # for seed, dataset, batch_size, strategy in configs:
    #     data, labels = odds_datasets.load(dataset)
    #     contamination_factor = np.sum(labels) / len(labels)
    #     run_sim(data, labels, batch_size, strategy, seed=seed, query_multiple=False, contamination_factor=contamination_factor)

    Parallel(n_jobs=-1)(delayed(run_sim)(odds_datasets.load(dataset)[0], 
                                         odds_datasets.load(dataset)[1], 
                                         batch_size, 
                                         strategy, 
                                         seed=seed, 
                                         query_multiple=False, 
                                         contamination_factor = np.sum(odds_datasets.load(dataset)[1]) / len(odds_datasets.load(dataset)[1]),
                                         dataset_name=dataset)
                                         for seed, dataset, batch_size, strategy in tqdm(configs))

if __name__ == "__main__":
    main()