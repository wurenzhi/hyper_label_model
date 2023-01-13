from collections import defaultdict
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import sys
import os
from baselines.baselines import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_dataset_wrench
from model import LELAWrapper


lela = LELAWrapper(checkpoint_path="lela_checkpoint.pt") #load pretrained LELA model


datasets = ["semeval","agnews", "trec", "spouse", "chemprot", "sms",  'census', 'commercial', 'youtube',
            "yelp", 'imdb', 'cdr', 'tennis', 'basketball']

def avg_scores(pct_adv_lf):
    rsts = defaultdict(list)
    for i in range(len(datasets)):
        X, y = load_dataset_wrench("datasets/"+datasets[i])

        # remove cols and rows with all abstentions
        non_zero_cols = np.sum(X >= 0, axis=0) != 0
        X = X[:, non_zero_cols]
        non_zero = np.sum(X >= 0, axis=1) != 0
        X = X[non_zero, :]
        y = y[non_zero]

        #inject adv LF
        adv_lf = np.zeros(shape=(X.shape[0],))-1
        #create a adv lf by flipping the first LF
        adv_lf[X[:, 0]==1] = 0 
        adv_lf[X[:, 0]==0] = 1
        #duplicate the adv lf multiple times
        n_adv_lf = int(pct_adv_lf*X.shape[1])
        X_adv = np.zeros(shape=(X.shape[0], n_adv_lf))-1
        for ll in range(n_adv_lf):
            X_adv[:, ll] = adv_lf
            X_adv[:, ll] = adv_lf
        X_adv = X_adv.astype(int)
        X = np.concatenate([X, X_adv], axis=1)

        y_ind = np.arange(0, X.shape[0])
        for method in [lela, Metal, dawid_skene, majority_vote, flying_squid, data_programming, ebcc,NPLM]:

            if isinstance(method, LELAWrapper):
                method_name = "LELA"
                pred_round = method.predict(X)
            else:
                method_name = method.__name__
                pred = method(X)
                pred_round = np.argmax(pred, axis=1).flatten()

            gt_labels = np.copy(y).flatten().astype(int)

            y_ind_ = y_ind[gt_labels >= 0]
            gt_labels = gt_labels[gt_labels >= 0]
            pred_round = pred_round[y_ind_]
            if datasets[i] in ["sms", "spouse", 'cdr',  'commercial', 'tennis', 'basketball',
                            'census']:
                acc = f1_score(gt_labels, pred_round)
            else:
                acc = accuracy_score(gt_labels, pred_round)
            rsts[method_name].append(acc)
    rsts_mean = {it:np.mean(rsts[it]) for it in rsts}
    print(pct_adv_lf, rsts_mean)
    return rsts_mean

rsts = defaultdict(list)
for pct_adv_lf in [0,0.1, 0.3, 0.5, 0.7, 0.9, 1]:
    rsts['num_adv_lf'].append(pct_adv_lf)
    score_dict = avg_scores(pct_adv_lf)
    for it in score_dict:
        rsts[it].append(score_dict[it])

rst_df = pd.DataFrame.from_dict(rsts)
rst_df.to_csv("results/adv_exp_acc.csv", index=False)
