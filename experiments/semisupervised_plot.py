from tkinter import font
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_csv('results/semisupervised_exp_acc.csv')
gp = data.groupby(["n","run"])
df = gp.mean()
X = [it[0] for it in list(df.index)]
X = list(sorted(set(X)))
lela_scores = [[df.loc[(n_ex,i_run),'lela']*100 for i_run in range(5)] for n_ex in X]
y_lela = [np.median(it) for it in lela_scores]
y_lela_max = [np.median(it)+np.std(it) for it in lela_scores]
y_lela_min = [np.median(it)-np.std(it) for it in lela_scores]

rf_scores = [[df.loc[(n_ex,i_run),'rf']*100 for i_run in range(5)] for n_ex in X]
y_rf = [np.median(it) for it in rf_scores]
y_rf_max = [np.median(it)+np.std(it) for it in rf_scores]
y_rf_min = [np.median(it)-np.std(it) for it in rf_scores]
fig = plt.figure(figsize=(5,3))
plt.axhline(69.0, color='black',ls="dotted", label='Unsupervised LELA')

plt.plot(X, y_lela, label='Semi-supervised LELA')

plt.fill_between(X, y_lela_min, y_lela_max,alpha=0.2)
plt.plot(X, y_rf, ls="--", label='Random Forest')

plt.fill_between(X, y_rf_min, y_rf_max,alpha=0.2)

plt.legend(loc='lower right')
plt.xscale('log')
plt.ylim([60, 71])
plt.xlabel(r"$N_{gt}$ number of gt labels",fontsize=13)
plt.ylabel("Averaged score over datasets",fontsize=11)
plt.tight_layout()
plt.savefig('results/semisupervised_fig.png', dpi=300)