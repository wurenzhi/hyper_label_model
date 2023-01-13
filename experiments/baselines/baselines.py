from .fast_dawid_skene.algorithms import run
from .ebcc import ebcc_predict
import numpy as np
from flyingsquid.label_model import LabelModel as fs_LM
from .data_programming import SrcGenerativeModel
from .nplm.noisy_partial import PartialLabelModel
import torch
import random


def majority_vote(l_matrix):
    n_class = np.max(l_matrix)+1
    n, m = l_matrix.shape
    Y_p = np.zeros((n, n_class))
    for i in range(m):
        for j in range(n_class):
            Y_p[:, j] = Y_p[:, j]+(l_matrix[:, i] == j)
    Y_p /= Y_p.sum(axis=1, keepdims=True)
    return Y_p

def NPLM(l_matrix):
    num_classes = np.max(l_matrix)+1
    label_partition = {i:[[j+1] for j in range(num_classes)] for i in range(l_matrix.shape[1])}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = PartialLabelModel(num_classes=num_classes,
                          label_partition=label_partition,
                          preset_classbalance=None,
                          device=device)
    model.optimize(l_matrix+1)
    labels = model.weak_label(l_matrix+1)
    labels = np.array(labels)
    parameters = list(model.parameters())
    del parameters
    torch.set_default_dtype(torch.float32)
    return labels

def dawid_skene(l_matrix):
    n_class = np.max(l_matrix)+1
    counts = np.zeros(
        shape=(l_matrix.shape[0], l_matrix.shape[1], n_class))
    for i in range(0, n_class):
        counts[l_matrix == i, i] = 1
    y = run(counts=counts, alg="DS")
    y_pred = np.zeros(shape=(l_matrix.shape[0], n_class))
    for i in range(n_class):
        y_pred[:, i] = (y == i).astype(int)
    return y_pred.astype(float)


def flying_squid(l_matrix):
    n_class = np.max(l_matrix)+1
    probs = []
    for i in range(n_class):
        fs_label_model = fs_LM(l_matrix.shape[1],triplet_seed=random.randint(0,1000000))
        target_mask = l_matrix == i
        abstain_mask = l_matrix == -1
        other_mask = (~target_mask) & (~abstain_mask)
        X_i = np.copy(l_matrix)
        X_i[target_mask] = 1
        X_i[abstain_mask] = 0
        X_i[other_mask] = -1
        fs_label_model.fit(X_i)
        preds_prob_fs = fs_label_model.predict_proba(X_i)[:, 1]
        probs.append(preds_prob_fs)
    probs = np.array(probs).T
    probs = probs/np.sum(probs, axis=1).reshape(-1, 1)
    return probs


def data_programming(l_matrix):
    n_class = np.max(l_matrix)+1
    probs = []
    for i in range(n_class):
        dp = SrcGenerativeModel(seed=random.randint(0,1000000))
        target_mask = l_matrix == i
        abstain_mask = l_matrix == -1
        other_mask = (~target_mask) & (~abstain_mask)
        X_i = np.copy(l_matrix)
        X_i[target_mask] = 1
        X_i[abstain_mask] = 0
        X_i[other_mask] = -1
        dp.train(X_i)
        preds_prob_fs = dp.predict_proba(X_i)[:, 1]
        probs.append(preds_prob_fs)
    probs = np.array(probs).T
    probs = probs/np.sum(probs, axis=1).reshape(-1, 1)
    return probs


def Metal(l_matrix):
    from snorkel.labeling.model import LabelModel as sk_LM
    l_matrix = np.copy(l_matrix)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sk_label_model = sk_LM(cardinality=np.max(l_matrix)+1, verbose=False,device=device)
    sk_label_model.fit(l_matrix, progress_bar=False,seed=random.randint(0,1000000))
    preds_prob_sk = sk_label_model.predict_proba(l_matrix)
    return preds_prob_sk


def ebcc(l_matrix):
    preds_ebcc = ebcc_predict(l_matrix)
    return preds_ebcc
