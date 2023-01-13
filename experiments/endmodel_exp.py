import copy
import logging
import os
import sys
from os.path import join

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight

from wrench._logging import LoggingHandler
from wrench.dataset import NumericDataset, RelationDataset, TextDataset
from wrench.dataset import load_dataset as wrench_load_dataset
from wrench.dataset import numeric_datasets, text_datasets
from wrench.endmodel import EndClassifierModel
from wrench.labelmodel import MajorityVoting, Snorkel
from model import LELAWrapper

def combine_wrench_train_and_valid(train_data, valid_data, dataset_cls):
    train_size = len(train_data)

    train_valid_data = dataset_cls()

    train_valid_data.ids = train_data.ids + [
        str(int(id) + train_size) for id in valid_data.ids
    ]
    train_valid_data.labels = train_data.labels + valid_data.labels
    train_valid_data.examples = train_data.examples + valid_data.examples
    train_valid_data.weak_labels = train_data.weak_labels + valid_data.weak_labels

    train_valid_data.n_class = valid_data.n_class
    train_valid_data.n_lf = valid_data.n_lf

    train_valid_data.features = np.concatenate(
        (train_data.features, valid_data.features), axis=0
    )
    train_valid_data.id2label = copy.deepcopy(train_data.id2label)

    return train_valid_data


def shuffle_and_split_train_valid(train_valid_data, dataset_cls, n_valid=None):
    N = len(train_valid_data)
    # Set the seed
    np.random.seed(79)
    shuffled_idx = np.arange(N)
    np.random.shuffle(shuffled_idx)
    # Revert the seed to a random one
    np.random.seed()

    train_valid_data.labels = np.array(train_valid_data.labels)[shuffled_idx].tolist()
    train_valid_data.examples = np.array(train_valid_data.examples)[
        shuffled_idx
    ].tolist()
    train_valid_data.weak_labels = np.array(train_valid_data.weak_labels)[
        shuffled_idx
    ].tolist()
    train_valid_data.features = train_valid_data.features[shuffled_idx]

    new_train_data = dataset_cls()
    new_valid_data = dataset_cls()

    n_valid = N // 4 if not n_valid else n_valid
    n_train = N - n_valid

    new_train_data.ids = [str(i) for i in range(n_train)]
    new_train_data.labels = train_valid_data.labels[:n_train]
    new_train_data.examples = train_valid_data.examples[:n_train]
    new_train_data.weak_labels = train_valid_data.weak_labels[:n_train]
    new_train_data.features = train_valid_data.features[:n_train, :]
    new_train_data.n_class = train_valid_data.n_class
    new_train_data.n_lf = train_valid_data.n_lf

    new_valid_data.ids = [str(i) for i in range(n_valid)]
    new_valid_data.labels = train_valid_data.labels[-n_valid:]
    new_valid_data.examples = train_valid_data.examples[-n_valid:]
    new_valid_data.weak_labels = train_valid_data.weak_labels[-n_valid:]
    new_valid_data.features = train_valid_data.features[-n_valid:, :]
    new_valid_data.n_class = train_valid_data.n_class
    new_valid_data.n_lf = train_valid_data.n_lf

    return train_valid_data, new_train_data, new_valid_data

def filter_abs_lfs(data):
    weak_labels = np.array(data.weak_labels)
    kept_lfs_idx = np.any(weak_labels != -1, axis=0)
    weak_labels = weak_labels[:, kept_lfs_idx]
    data.weak_labels = weak_labels.tolist()
    data.n_lf = len(kept_lfs_idx)
    return data

def aggregate_results():
    with open(join("results", "em_grid_agg.csv"), "a") as f:
        f.write("model," + ",".join(datasets_and_metrics.keys()) + ",mean,var\n")

    for model in label_models:
        trial_maxes = []
        for data in datasets_and_metrics.keys():
            csv_path = join(dataset_path, "em_grid_{}_{}.csv".format(data, model))

            if data in numeric_datasets:
                names = [
                    "dataset",
                    "trial",
                    "batch_size",
                    "lr",
                    "wd",
                    "valid_score",
                    "test_score",
                ]
            else:
                names = [
                    "dataset",
                    "trial",
                    "batch_size",
                    "lr",
                    "valid_score",
                    "test_score",
                ]

            df = pd.read_csv(csv_path, names=names)

            trial_max = df.loc[df.groupby(["trial"])["valid_score"].idxmax()]
            trial_max = trial_max[["trial", "test_score"]]
            trial_max.insert(0, "model", model)
            trial_max.insert(0, "dataset", data)
            trial_maxes.append(trial_max)

        trial_maxes = pd.concat(trial_maxes)
        mean = trial_maxes.groupby(by="trial").mean()["test_score"].mean()
        var = (
            trial_maxes.groupby(by="trial").mean()["test_score"].max()
            - trial_maxes.groupby(by="trial").mean()["test_score"].min()
        ) / 2
        result = trial_maxes.groupby(by="dataset").mean()["test_score"]
        result = [str(round(result[data], 4)) for data in datasets_and_metrics.keys()]
        result = [model] + result + [str(mean), str(var)]
        with open(join("results", "endmodel_exp_acc.csv"), "a") as f:
            f.write(",".join(result) + "\n")


datasets_and_metrics = {
    "semeval": "acc",
    "agnews": "acc",
    "trec": "acc",
    "spouse": "f1_binary",
    "chemprot": "acc",
    "sms": "f1_binary",
    "census": "f1_binary",
    "commercial": "f1_binary",
    "youtube": "acc",
    "yelp": "acc",
    "imdb": "acc",
    "cdr": "f1_binary",
    "tennis": "f1_binary",
    "basketball": "f1_binary",
}

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

logger = logging.getLogger(__name__)

device = "cuda:0"


label_models = ["mv", "snorkel", "lela"] #snorkel is the same as MeTal
dataset_path = "./datasets"

for label_model_name in label_models:
    for data, metric in datasets_and_metrics.items():
        dataset_cls = None
        if data in numeric_datasets:
            dataset_cls = NumericDataset
        elif data in text_datasets:
            dataset_cls = TextDataset
        else:
            dataset_cls = RelationDataset

        for trial in range(10):
            # Load dataset data
            train_data, valid_data, test_data = wrench_load_dataset(
                dataset_path,
                data,
                extract_feature=True,
                extract_fn="bert",  # extract bert embedding
                model_name="bert-base-cased",
                cache_name="bert",
                device=device,
            )
            # Filter out uncovered training data
            train_data = train_data.get_covered_subset()
            valid_data = valid_data.get_covered_subset()
            test_data = test_data.get_covered_subset()

            # Combine training and validation data and re-split
            train_valid_data = combine_wrench_train_and_valid(
                train_data, valid_data, dataset_cls
            )
            train_valid_data, train_data, valid_data = shuffle_and_split_train_valid(
                train_valid_data, dataset_cls
            )

            # Filter out LFs that abstains on all data entries
            train_valid_data = filter_abs_lfs(train_valid_data)
            test_data = filter_abs_lfs(test_data)

            # Get balance for Snorkel
            n_class = valid_data.n_class
            balance = np.ones(n_class) / n_class

            if label_model_name == "lela":
                # Load dataset LFs (training split + valid split)
                lfs = np.array(train_valid_data.weak_labels)
                # lfs = np.array(train_valid_test_data.weak_labels)

                # Load label model: our model
                
                label_model = LELAWrapper(checkpoint_path="lela_checkpoint.pt")
                aggregated_soft_labels = label_model.predict_prob(lfs)
                aggregated_hard_labels = np.argmax(
                    aggregated_soft_labels, axis=1
                ).flatten()

            elif label_model_name == "snorkel":
                # Load label model: Snorkel
                label_model = Snorkel()
                # Train label model
                label_model.fit(dataset_train=train_valid_data, balance=balance)
                # Run label model
                aggregated_hard_labels = label_model.predict(train_valid_data)
                aggregated_soft_labels = label_model.predict_proba(train_valid_data)

            elif label_model_name == "mv":
                # Load label model: Majority Voting
                label_model = MajorityVoting()
                # Train label model
                label_model.fit(dataset_train=train_valid_data)
                # Run label model
                aggregated_hard_labels = label_model.predict(train_valid_data)
                aggregated_soft_labels = label_model.predict_proba(train_valid_data)

            agg_train_soft_labels = aggregated_soft_labels[: len(train_data), :]
            agg_train_hard_labels = aggregated_hard_labels[: len(train_data)]
            agg_valid_hard_labels = aggregated_hard_labels[-len(valid_data) :]
            valid_data.labels = (
                agg_valid_hard_labels.tolist()
            )  # Use the labels predicted by label model

            labels, label_weights = np.unique(agg_train_hard_labels, return_counts=True)
            label_weights = label_weights / len(agg_train_hard_labels)
            label_weights = {c: w for c, w in zip(labels, label_weights)}
            sample_weights = compute_sample_weight(label_weights, agg_train_hard_labels)

            if dataset_cls == NumericDataset:
                # Grid Search
                for batch_size in [512, 128, 32]:
                    for lr in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
                        for wd in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
                            ffn_num_layer = 2
                            ffn_hidden_size = 100

                            # Run end model: MLP
                            model = EndClassifierModel(
                                batch_size=batch_size,
                                test_batch_size=512,
                                n_steps=10000,
                                backbone="MLP",
                                optimizer="Adam",
                                optimizer_lr=lr,
                                optimizer_weight_decay=wd,
                                n_hidden_layers=ffn_num_layer,
                                hidden_size=ffn_hidden_size,
                            )
                            model.fit(
                                dataset_train=train_data,
                                y_train=agg_train_hard_labels,
                                dataset_valid=valid_data,
                                evaluation_step=10,
                                metric=metric,
                                patience=100,
                                device=device,
                            )
                            valid_score = model.test(valid_data, metric)
                            test_score = model.test(test_data, metric)
                            logger.info(
                                "End model (MLP) hyperparams: bs = {}, lr = {}, wd = {}, ffn_num_layer = {}, ffn_hidden_size = {}".format(
                                    batch_size, lr, wd, ffn_num_layer, ffn_hidden_size
                                )
                            )
                            logger.info(
                                "End model (MLP) test {}: {}".format(
                                    metric, valid_score
                                )
                            )
                            logger.info(
                                "End model (MLP) test {}: {}".format(metric, test_score)
                            )

                            with open(
                                "./datasets/datasets/em_grid_{}_{}.csv".format(
                                    data, label_model_name
                                ),
                                "a",
                            ) as f:
                                f.write(
                                    "{},{},{},{},{},{},{}\n".format(
                                        data,
                                        trial,
                                        batch_size,
                                        lr,
                                        wd,
                                        valid_score,
                                        test_score,
                                    )
                                )

            else:
                # Grid Search
                for batch_size in [32, 16]:
                    for lr in [0.00002, 0.00003, 0.00005]:
                        # Run end model: BERT
                        model = EndClassifierModel(
                            batch_size=32,
                            real_batch_size=batch_size,  # for accumulative gradient update
                            test_batch_size=64,
                            n_steps=1000,
                            backbone="BERT",
                            backbone_model_name="bert-base-cased",
                            backbone_max_tokens=128,
                            backbone_fine_tune_layers=-1,  # fine  tune all
                            optimizer="AdamW",
                            optimizer_lr=lr,
                            optimizer_weight_decay=0.0,
                        )
                        model.fit(
                            dataset_train=train_data,
                            y_train=agg_train_soft_labels,
                            dataset_valid=valid_data,
                            evaluation_step=10,
                            metric=metric,
                            patience=50,
                            device=device,
                        )
                        valid_score = model.test(valid_data, metric)
                        test_score = model.test(test_data, metric)
                        logger.info(
                            "End model (BERT) hyperparams: bs = {}, lr = {}".format(
                                batch_size, lr
                            )
                        )
                        logger.info(
                            "End model (BERT) test {}: {}".format(metric, valid_score)
                        )
                        logger.info(
                            "End model (BERT) test {}: {}".format(metric, test_score)
                        )

                        with open(
                            "./datasets/datasets/em_grid_{}_{}.csv".format(
                                data, label_model_name
                            ),
                            "a",
                        ) as f:
                            f.write(
                                "{},{},{},{},{},{}\n".format(
                                    data, trial, batch_size, lr, valid_score, test_score
                                )
                            )

aggregate_results()