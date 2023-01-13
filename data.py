import math
import time
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from os.path import join
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import coo_matrix


def load_dataset_wrench(path):
    """load datasets provided in the wrench project
    please download from https://github.com/JieyuZ2/wrench
    """
    import json
    dicts = []
    for name in ["train.json", "valid.json", "test.json"]: #some datasets miss one file
        try:
            dict = json.load(open(join(path, name)))
            dicts.append(dict)
        except:
            pass
    if len(dicts)==0:
        raise Exception("error while loading datasets!")
    data_dict = {}
    for d in dicts:
        for k, v in d.items():  
            data_dict[k] = v
    lfs = []
    ys = []
    for i in data_dict:
        ys.append(data_dict[i]['label'])
        lfs.append(data_dict[i]['weak_labels'])
    LF_matrix = np.array(lfs)
    y = np.array(ys)
    return LF_matrix, y


def generate_independent_lf(n_lfs=50):
    """Generate sythetic data based on the conditional independence assumption. 
       This is used to generate the synthetic validation set.
    """
    n_example = 10000  
    pos_ratio = random.random()
    gt = np.zeros(n_example)
    n_pos = int(pos_ratio * n_example)
    n_neg = n_example - n_pos
    gt[:n_pos] = 1
    LFs = []
    n_dummy_lf = int(random.random()*n_lfs*0.8)
    pos_max_precision_range = 0.5*random.random()
    neg_max_precision_range = 0.5*random.random()
    for i in range(n_lfs):
        if i < n_dummy_lf:
            continue
        else:
            pos_coverage = random.random()
            neg_coverage = random.random()
            pos_precision = 0.5 + random.random()*pos_max_precision_range
            neg_precision = 0.5 + random.random()*neg_max_precision_range
            lf = np.zeros(n_example)
            covered_pos = np.random.choice(
                n_pos, int(pos_coverage*n_pos), replace=False)
            tp = np.random.choice(covered_pos, int(
                pos_precision*pos_coverage*n_pos), replace=False)
            lf[covered_pos] = -1
            lf[tp] = 1
            covered_neg = n_pos + \
                np.random.choice(n_neg, int(
                    neg_coverage * n_neg), replace=False)
            tn = np.random.choice(covered_neg, int(
                neg_precision*neg_coverage*n_neg), replace=False)
            lf[covered_neg] = 1
            lf[tn] = -1
        LFs.append(lf)
    LFs = np.array(LFs).T
    return LFs, gt


def try_replace(X, gt):
    """Replace with a new pair if the weaker form of the better-than-random assumption is not satisfied.
       In order to make the new pair has the same number of abstentions (so we can train batch by batch), we construct a new pair based on the pair X, gt.
       Based on Lemma 2, it is easy to see the construction in this function also ensures p(y|X) to be uniform.
    Args:
        X: label matrix
        gt: ground-truth labels

    Returns:
        A pair of (X, gt) that satisfies the better-than-random assumption.
    """
    X = np.array(X)
    gt = np.squeeze(gt)
    n_lfs = X.shape[1]
    for gt_label in [1, 0]:
        n_worse_than_random = 0
        for i_lf in range(n_lfs):
            if gt_label == 0:
                if np.sum(X[gt == 0, i_lf] == 1) > np.sum(X[gt == 0, i_lf] == -1):
                    n_worse_than_random += 1
            else:
                if np.sum(X[gt == 1, i_lf] == -1) > np.sum(X[gt == 1, i_lf] == 1):
                    n_worse_than_random += 1
        if n_worse_than_random >= n_lfs/2:
            for i_lf in range(n_lfs):
                if gt_label == 0:
                    X[gt == 0, i_lf] = -X[gt == 0, i_lf]
                else:
                    X[gt == 1, i_lf] = -X[gt == 1, i_lf]
    return X, gt


class DatasetOnlineGen(Dataset):
    """Sythetical Dataset used for training, data points are generated on-the-fly
    """
    def __init__(self, size, max_n_lfs=60, max_example=2000):
        self.size = size # this is just to trick data loader to work, the dataset is generated on the fly so there is no dataset size
        self.max_n_lfs = max_n_lfs
        self.max_example = max_example
        self.random = random.Random(time.time())

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        #randomly generate a label matrix X_max on-the-fly of maximum size
        #the X_max matrix is then down-sampled during collate
        gt = (np.random.rand(self.max_example) > 0.5).astype(int)
        LFs = np.random.rand(self.max_example, self.max_n_lfs)
        LFs[LFs > 2/3] = 1
        LFs[LFs < 1/3] = -1
        LFs = LFs.astype(int)
        return LFs, gt

    def collate(self, batch):
        """ Custom collation on CPU. 
        down-sample batch matrices and obtain sparse representation of the label matrices
        """
        # down-sample each data in the
        # batch to size (n_example, n_lfs) with random example and LF choices.
        # This allows each different batches to contain different sizes of matrices.
        # e.g. batch_1.shape = (50, 350, 24), batch_2.shape = (50, 128, 42)
        n_lfs = self.random.randint(2, self.max_n_lfs)
        #n_example = int(10 ** (self.random.random()
        #                    * (math.log10(self.max_example) - 2) + 2))
        n_example = self.random.randint(100, self.max_example)

        # If matrix is too large, it causes CUDA out of memory.
        if n_lfs*n_example > 50*2000:
            n_example = int(50*2000/n_lfs)

        batch_X, batch_y = [], []
        n_abstain_entries = []
        for data in batch:
            X, y = data

            # Sample indices based on n_example and n_lfs
            lf_indices = np.random.choice(self.max_n_lfs, size=n_lfs, replace=False)
            example_indices = np.random.choice(self.max_example, size=n_example, replace=False)

            # Downsample X and y via indexing
            X, y = X[np.ix_(example_indices, lf_indices)], y[example_indices]

            batch_X.append(X)
            batch_y.append(y)
            n_abstain_entries.append(np.sum(X == 0))

        # obtain the sparse representation of the matrices in the same batch
        # In order to feed the sparse matrices in one batch,
        # the number of abstention (or equivalently non-abstention) elements should be the same for all matrices in the same batch
        n_abs = random.choice(n_abstain_entries) # specify the number of abstention elements for each matrix in the batch
        if n_abs == n_example*n_lfs:
            n_abs = np.min(n_abstain_entries)

        batch_index_sparse, batch_value_sparse, batch_y_sparse = [], [], []
        for i in range(len(batch_X)): # obtain the sparse representation for all matrices
            X, y = batch_X[i], batch_y[i]
            n_abs_i = np.sum(X == 0)
            if n_abs_i < n_abs: # If the number of abstention elements is smaller than the desire n_abs, randomly drop some elements
                X_sparse = coo_matrix(X)
                index = np.array([X_sparse.row, X_sparse.col]).T
                value = X_sparse.data
                keep = np.random.choice(
                    index.shape[0], n_example*n_lfs-n_abs, replace=False)
                keep = np.sort(keep)
                index = index[keep, :]
                value = value[keep]
            else: # If the number of abstention elements is larger than the desire n_abs, randomly add some elements
                if n_abs_i > n_abs:
                    zero_entries = np.argwhere(X == 0)
                    to_change = np.random.choice(
                        zero_entries.shape[0], n_abs_i-n_abs, replace=False)
                    for i in range(to_change.shape[0]):
                        X[zero_entries[to_change[i]][0], zero_entries[to_change[i]]
                            [1]] = 1 if i % 2 == 0 else -1
                X_sparse = coo_matrix(X)
                index = np.array([X_sparse.row, X_sparse.col]).T
                value = X_sparse.data

            X_sparse = coo_matrix(
                (np.squeeze(value), (index[:, 0], index[:, 1])), shape=X.shape)
            X = X_sparse.todense()
            X, y = try_replace(X, y) #replace with a new pair if the better than random assumption is not satisfied
            X_sparse = coo_matrix(X)
            index = np.array([X_sparse.row, X_sparse.col]).T
            value = X_sparse.data
            batch_index_sparse.append(index)
            batch_value_sparse.append(value)
            batch_y_sparse.append(y)
        batch_index_sparse = np.stack(batch_index_sparse)
        batch_value_sparse = np.stack(batch_value_sparse)
        batch_y_sparse = np.stack(batch_y_sparse)
        batch_index_sparse, batch_value_sparse, batch_y_sparse = torch.from_numpy(
            batch_index_sparse), torch.from_numpy(batch_value_sparse), torch.from_numpy(batch_y_sparse)
        return batch_index_sparse, batch_value_sparse, batch_y_sparse



class SytheticValidation:
    """helper class that contains the sythetically generated validation set
    """
    def __init__(self) -> None:
        self.Xy_synthetic = []
        # fix random seed for generating the sythetic validation set,
        # so that different runs use the same validation set 
        # and we can select the best run based on validation acc 
        random.seed(0) 
        np.random.seed(0)
        for i in range(100):
            X, y = generate_independent_lf()
            non_zero_cols = np.sum(X >= 0, axis=0) != 0
            X = X[:, non_zero_cols]
            # non zero rows
            non_zero = np.sum(X >= 0, axis=1) != 0
            X = X[non_zero, :]
            y = y[non_zero]
            self.Xy_synthetic.append((X, y))
        random.seed(int(time.time()))
        np.random.seed(int(time.time()))
        print(np.sum(self.Xy_synthetic[0][0]), np.sum(self.Xy_synthetic[0][1]))

    def get_avg_score_sythetic(self, model):
        scores = []
        for i in range(len(self.Xy_synthetic)):
            X, y = self.Xy_synthetic[i]
            inputs, labels = torch.from_numpy(np.expand_dims(
                X, 0)), torch.from_numpy(np.expand_dims(y, 0))
            pred = pred_binary_class(model, inputs)
            pred = pred.detach().cpu().numpy()
            pred_round = np.array(pred > 0.5).astype(int)
            gt_labels = labels.numpy().flatten().astype(int)
            pred_round = pred_round[gt_labels >= 0]
            gt_labels = gt_labels[gt_labels >= 0]
            acc = accuracy_score(gt_labels, pred_round)
            scores.append(acc)
        return np.mean(scores)

def pred_binary_class(model, LF_mat):
    device = next(model.parameters()).device
    LF_mats = np.array(LF_mat)
    X_sparse = coo_matrix(np.squeeze(LF_mats))
    index = np.array([X_sparse.row, X_sparse.col]).T
    value = X_sparse.data
    index = torch.from_numpy(index).to(device)
    value = torch.from_numpy(value).float().to(device)
    pred, _ = model(index.unsqueeze(0), value.unsqueeze(0))
    return pred
