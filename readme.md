# Implementation Learning Hyper Label Model for Programmatic Weak Supervision (https://arxiv.org/abs/2207.13545)

** IMPORTANT NOTE: we only tested the instructions in this file on an ubuntu machine. If you encounters problems with Windows or Mac, please switch to an ubuntu machine. 

# 1. Download Datasets
Please download the 14 classification datasets provided by the WRENCH project https://github.com/JieyuZ2/wrench and put them in the datasets/ folder.

# 2. Create Conda Environment
```
conda env create -f environment.yml
conda activate LELA_exps
conda install numba
```
If you encounter errors later when runing the scripts for the experiments, please try:
```
conda install numba==0.55.1
```


# 3. Reproducing Experiment Results
In this section, we show how to quickly reproduce the numbers and figures in our paper using our provided trained LELA model. We will show how to train LELA from scratch in the next section.

## 3.1. Label Aggregation Performance & Running Time
```
python experiments/performance_exp.py
```
This generates a performance_exp_acc.csv file for the performance scores and a performance_exp_time.csv file for running times in the results/ folder. 
## 3.2. Semi-supervised Label Aggregation
```
python experiments/semisupervised_exp.py
```
This generates a semisupervised_exp_acc.csv file in the results/ folder. 

To plot Figure 2:
```
python experiments/semisupervised_plot.py
```
This generates a figure semisupervised_fig.png in the results/ folder.
## 3.3. End-model Performance
```
python experiments/endmodel_exp.py
```
This generates a endmodel_exp_acc.csv file in the results/ folder. 

# 4. Training LELA
If you don't want to use our provided trained model for LELA, you can train LELA from scratch by:
```
python train.py
```
This trains the LELA model 10 runs and prints the checkpoint of the run selected based on accuracy of the synthetic validation set. The checkpoints can be found in the model_checkpoints/ folder.
Please use the suggested checkpoint for your evaluation.

Note training can take 2-3 days. To speedup, one can adapt the script to use a different GPU for a different run to execute the runs in parallel.

# 5. LICENSE
Our code is released under GNU GENERAL PUBLIC LICENSE Version 3. See the LICENSE file.
