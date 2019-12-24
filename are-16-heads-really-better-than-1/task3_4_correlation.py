import re
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import pearsonr


taskx = 'CoLA' # task displayed on x axis
tasky = 'SST2' # task displayed on y axis

# prepare data for x axis

f = open(taskx+"out.txt", 'r', errors='ignore')
lines = f.readlines()
f.close()

base_accx = float(lines[-3].split()[-1])
ablation_linesx = lines[-1].split('Layer:')
for i, layer in enumerate(ablation_linesx):
	ablation_linesx[i] = layer.split('\t')[1:]
ablation_linesx = ablation_linesx[1:]
# convert from str to float
for i in range(len(ablation_linesx)):
	for j in range(len(ablation_linesx[i])):
		ablation_linesx[i][j] = float(ablation_linesx[i][j])

ablation_linesx = np.array(ablation_linesx).flatten() + base_accx
print(ablation_linesx)

# prepare data for y axis
f = open(tasky+"out.txt", 'r', errors='ignore')
lines = f.readlines()
f.close()

base_accy = float(lines[-3].split()[-1])
ablation_linesy = lines[-1].split('Layer:')
for i, layer in enumerate(ablation_linesy):
	ablation_linesy[i] = layer.split('\t')[1:]
ablation_linesy = ablation_linesy[1:]
# convert from str to float
for i in range(len(ablation_linesy)):
	for j in range(len(ablation_linesy[i])):
		ablation_linesy[i][j] = float(ablation_linesy[i][j])

ablation_linesy = np.array(ablation_linesy).flatten() + base_accy
print(ablation_linesy)

r_score, _ = pearsonr(ablation_linesx, ablation_linesy)
print("r_score;", r_score)

fig, ax = plt.subplots()
plt.scatter(ablation_linesx, ablation_linesy, marker = '+', label='Head ablation accuracies (Pearson r = '+ str(round(r_score*100)/100)+')')
plt.plot([base_accx], [base_accy], marker='*', markersize=7, color="red", label='Original accuracies')
#plt.scatter(base_accx, base_accy, color='r', markersize=12)
leg=ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
plt.xlabel(taskx, fontsize=20)
plt.ylabel(tasky, fontsize=20)
plt.show()







