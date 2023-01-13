import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('results/robustness_exp_acc.csv')
data.columns = ['num_adv_lf','LELA','MeTaL','DS','MV','FS','DP','EBCC',"NPLM"]
data.index = data['num_adv_lf']
methods = ["MV","DP","FS","MeTaL","NPLM","DS",'EBCC',"LELA"]
fig = plt.figure(figsize=(5,3))

lss = ['dashed', 'dotted','dotted','dashdot','dotted','dotted','dotted', 'solid']
mss = ['^','.','.','x', '.','.','.','*']

for i in range(len(methods)):
    mtd = methods[i]
    if mtd == "LELA":
        plt.plot(data.index, data[mtd]*100,ls = lss[i],marker=mss[i], ms=8,label=mtd)
    else:
        plt.plot(data.index, data[mtd]*100,ls = lss[i],marker=mss[i], label=mtd)

plt.legend(title='methods', bbox_to_anchor=(1, 1))
#data.plot(figsize=(5,3),style='.-').legend(title='Methods', bbox_to_anchor=(1, 1))
plt.xlabel(r"$\beta$ amount of adversarial LF duplicates")
plt.ylabel("Averaged score over all datasets")
plt.tight_layout()
plt.savefig('results/robustness_fig.png', dpi=300)
#plt.ylim([0,20])