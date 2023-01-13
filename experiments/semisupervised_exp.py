from collections import defaultdict
import time
import torch
import torch.optim as optim
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import os
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import LELASemi
from tqdm import tqdm

from data import load_dataset_wrench


def random_forest(LF_mat, y_partial, y_indices):
    from sklearn.model_selection import GridSearchCV
    param_grid = {'max_depth': [2, 4, 8, 16, 32, None],
                  'min_samples_split': [2, 5],
                  }
    try:
        clf = GridSearchCV(RandomForestClassifier(
            n_estimators=500, random_state=0, class_weight="balanced"), param_grid, n_jobs=-1)
        clf.fit(LF_mat[y_indices,:], y_partial)
    except:# when the number of data points is too small, cross_validation throws error
        clf = RandomForestClassifier(
            n_estimators=500, random_state=0, class_weight="balanced")
        clf.fit(LF_mat[y_indices,:], y_partial)
    pred = clf.predict_proba(LF_mat)
    return pred

final_rsts = defaultdict(list)
nums = [10,20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
n_run = 5
for n_gt in tqdm(nums):
    for i_run in range(n_run):
        datasets = ["semeval", "agnews", "trec", "spouse", "chemprot", "sms",  'census', 'commercial', 'youtube',
                    "yelp", 'imdb', 'cdr', 'tennis', 'basketball']
        for i in range(len(datasets)):
            final_rsts['n'].append(n_gt)
            final_rsts['run'].append(i_run)
            final_rsts['dataset'].append(datasets[i])
            X, y = load_dataset_wrench("datasets/"+datasets[i])

            # remove cols and rows with all abstentions
            non_zero_cols = np.sum(X >= 0, axis=0) != 0
            X = X[:, non_zero_cols]
            non_zero = np.sum(X >= 0, axis=1) != 0
            X = X[non_zero, :]
            y = y[non_zero]

            y_ind = np.arange(0, X.shape[0])

            y_indices = []
            non_abs_locs = np.where(y>=0)[0]
            # select at most 70% of the labels in order to have a reliable test score on the remaining data points
            y_indices=np.random.choice(non_abs_locs,min(n_gt,int(len(non_abs_locs)*0.7)),replace=False) 
            for i_class in range(np.max(y)+1):
                # make sure each class has at least two data points
                if np.sum(y[y_indices]==i_class)==0:
                    class_locs = np.where(y==i_class)[0]
                    y_indices = np.concatenate([y_indices, np.random.choice(class_locs,2,replace=False)])
            y_partial = y[y_indices]
            
            for method_name in ["lela","rf"]:
                t = time.time()
                if method_name == "lela":
                    pred = LELASemi(X, y_partial, y_indices,checkpoint_path="lela_checkpoint.pt")
                    pred = pred.detach().cpu().numpy()
                else:
                    pred = random_forest(X, y_partial, y_indices)
                    
                pred_round = np.argmax(pred, axis=1).flatten()
                gt_labels = np.copy(y)
                keep = gt_labels >= 0
                keep[y_indices] = False
                y_ind_ = y_ind[keep]
                gt_labels = gt_labels[keep]
                pred_round = pred_round[y_ind_]
                if datasets[i] in ["sms", "spouse", 'cdr',  'commercial', 'tennis', 'basketball',
                                'census']:
                    acc = f1_score(gt_labels, pred_round)
                else:
                    acc = accuracy_score(gt_labels, pred_round)
                final_rsts[method_name].append(acc)

rst_df = pd.DataFrame.from_dict(final_rsts)
rst_df.to_csv("results/semisupervised_exp_acc.csv", index=False)
