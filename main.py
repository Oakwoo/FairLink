from utils.parameter_parser import parameter_parser
from utils.utils import tab_printer, read_data
from algs.proximity_base import jaccard, adamic_adar, preferential_attachment
from algs.mab import MAB_LP

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import time
from datetime import date
from utils.eval import compute_unfairness
from sklearn.model_selection import GridSearchCV




def main():
    args = parameter_parser()
    tab_printer(args)
    data = read_data(args)
    train, test = train_test_split(data['examples'], test_size=args.test_size, random_state=1)
    G = data['G']
    communities = data['communities']
    for i in range(len(communities)):
        for node in communities[i]:
            G.nodes[node]['att'] = i

    unfairness_ground = compute_unfairness(G,communities)

    add_radio = args.add_radio
    part_size = int(len([i for i in test if i[2]==1]) * add_radio)
    print("try to add ",int(part_size)," edges")
    batch_step = args.batch_step

    # BASELINE EXPERIMENT
    print("Baseline start")
    G_baseline = G.copy()
    test_baseline = test.copy()
    TP_baseline = 0
    start_baseline = time.time()
    for iter in tqdm(range(int(part_size/batch_step))):
        if args.algorithm == "jac":
            y_pred = jaccard(G_baseline, test_baseline)
        elif args.algorithm == "adar":
            y_pred = adamic_adar(G_baseline, test_baseline)
        elif args.algorithm == "prf":
            y_pred = preferential_attachment(G_baseline, test_baseline)
        y_pred = list(zip(test_baseline, y_pred))
        y_pred = sorted(y_pred, key=lambda x:x[1], reverse=True)
        # select top batch_step edges
        test_baseline = np.array([left[0] for left in y_pred[batch_step:]]) # unzip predict score
        y_pred = y_pred[:batch_step]
        for edge in y_pred:
            if edge[0][2] == 1: # only add true, link prediction
                G_baseline.add_edge(edge[0][0], edge[0][1])
                TP_baseline += 1
        
    end_baseline = time.time()
    print("Baseline process time: {} seconds".format(end_baseline - start_baseline))
    # compute precision
    precision = TP_baseline / ((iter + 1) * batch_step)
    print("Baseline precision:", precision)
    unfairness_baseline = compute_unfairness(G_baseline,communities,output_detail=0)
    IU_baseline_improve = np.round((unfairness_ground - unfairness_baseline) / unfairness_ground, 4)
    print("Baseline information unfairness improve:", IU_baseline_improve)
    print("Baseline finished\n\n")

    # FairLink EXPERIMENT
    print("FairLink start")
    print("Grid searching best parameter...")
    parameters = {'gamma':[0.9, 0.92, 0.95, 0.98, 1]}
    MAB = GridSearchCV(MAB_LP(G.copy(), data['communities'], add_radio=args.add_radio,
                            batch_step=args.batch_step, acc_bound = args.acc_bound, 
                            gamma=args.gamma, slot_number = args.slot_number, e = args.epsilon,
                            decay = args.decay), parameters, cv=args.cross_validate)
    MAB.fit(train, train)
    print('Best parameter：', MAB.best_params_)
    print('Best score on validate set：', MAB.best_score_)
    best_model = MAB.best_estimator_
    start_MAB = time.time()
    best_model.predict(test)

    end_MAB = time.time()
    print("FairLink process time: {} seconds".format(end_MAB - start_MAB))

    unfairness_new = best_model.get_new_information_unfairness()
    unfairness = np.round((unfairness_ground- unfairness_new)/unfairness_ground, 4)
    acc = best_model.eval_accuracy()
    print('FairLink auc:', acc)
    print('FairLink information unfairness improve:', unfairness)



if __name__ =="__main__":
    main()
