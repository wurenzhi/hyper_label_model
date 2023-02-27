### Hyper label model
A hyper label model to aggregate weak labels from multiple weak supervision sources to infer the ground-truth labels in a single forward pass.

For more details, see our ICLR23 paper [Learning Hyper Label Model for Programmatic Weak Supervision](https://arxiv.org/abs/2207.13545)

** To reproduce experiments of our paper or to re-train the model from scratch, please switch to the paper_experiments branch.

### How to use
1. Install the package
   
    `pip install hyperlm`

2. Import and create an instance

```python
   from hyperlm import HyperLabelModel
   hlm = HyperLabelModel()
```
3. **Unsupervised label aggregation**. Given an weak label matrix `X`, e.g. `X=[[0, 0, 1],
                  [1, 1, 1],
                  [-1, 1, 0],
                  [0, 1, 0]]`, you can infer the labels by:
```python
   pred = hlm.infer(X)
```
Note in `X`, `-1` represents abstention,  `0` and `1` represent classes. Each row of `X` includes the weak labels for a data point, and each column of `X` includes the weak labels from a labeling function (LF).

4. **Semi-supervised label aggregation**. Let's say the gt labels are provided for the examples at index 1 and 3, i.e. `y_indices=[1,3]`, and the gt labels are `y_vals=[1, 0]`. We can incorporate the provided partial ground-truth with:

```python
   pred = hlm.infer(X, y_indices=y_indices,y_vals=y_vals)
```

### Citation
```
@inproceedings{
wu2023learning,
title={Learning Hyper Label Model for Programmatic Weak Supervision},
author={Renzhi Wu and Shen-En Chen and Jieyu Zhang and Xu Chu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=aCQt_BrkSjC}
}
```
