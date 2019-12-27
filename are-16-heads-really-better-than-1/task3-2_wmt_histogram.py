import re
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats

# read file
filename = 'wmt_en-fr_out2.txt'
f = open(filename, 'r', errors='ignore')
lines = f.readlines()
f.close()
baseacc = float(lines[0])
print(baseacc)
enen = lines[2:8]
ende = lines[9:15]
dede = lines[16:22]

# scrape head ablation scores
for i, layer in enumerate(enen):
	layer = layer.rstrip('\n')
	enen[i] = layer.split('\t')[1:]
for i, layer in enumerate(ende):
	ende[i] = layer.split('\t')[1:]
for i, layer in enumerate(dede):
	dede[i] = layer.split('\t')[1:]

# obtain accuracy by adding difference by base accuracy
enen = np.array(enen).astype(float) + baseacc
ende = np.array(ende).astype(float) + baseacc
dede = np.array(dede).astype(float) + baseacc

ablation_lines_flatten = list(enen.flatten()) + list(ende.flatten()) + list(dede.flatten())

accuracy_dist = {}
for acc in ablation_lines_flatten:
		key = math.floor((acc)*10000)/10000
		accuracy_dist[key] = accuracy_dist.get(key, 0) + 1

hist, bin_edges = np.histogram(ablation_lines_flatten, bins=len(accuracy_dist))
plt.figure(figsize = [8, 8])
plt.bar(bin_edges[:-1], hist, width=1, color='white')
plt.xlim(min(bin_edges), max(bin_edges))
plt.xlabel('BLEU', fontsize=20)
plt.ylabel('# Heads', fontsize=20)
plt.hist(ablation_lines_flatten, bins=len(hist))
plt.axvline(x=baseacc, color='r')
plt.tick_params(labelsize=17)
plt.show(block=True)



