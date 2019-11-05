import re
import matplotlib.pyplot as plt
import math
import numpy as np


# read file
f = open("MNLIout.txt", 'r', errors='ignore')
lines = f.readlines()
f.close()

# prepare data from file
num_example = None
evaluation_speed = []
for line in lines:
	if 'Num examples' in line:
		num_example = float(line.split()[-1])
	if line.startswith('Evaluating'):
		speed = re.findall(", (.*?)it/s", line)
		if speed:
			evaluation_speed.extend(speed)
evaluation_speed = evaluation_speed[1:]
base_acc = float(lines[-3].split()[-1])
ablation_lines = lines[-1].split('Layer:')
for i, layer in enumerate(ablation_lines):
	ablation_lines[i] = layer.split('\t')[1:]
ablation_lines = ablation_lines[1:]
#print("number of example:", num_example)
#print("batch size:", batch_size)
#print("speeds:", evaluation_speed)
#print("base accuracy:", base_acc)
#print("ablation line:", ablation_lines)

# convert from str to float
for i in range(len(ablation_lines)):
	for j in range(len(ablation_lines[i])):
		ablation_lines[i][j] = float(ablation_lines[i][j])

# build histogram from ablation accuracy
accuracy_dist = {}
for layer in ablation_lines:
	for head in layer:
		key = math.floor((base_acc+ head)*10000)/10000
		accuracy_dist[key] = accuracy_dist.get(key, 0) + 1
#print(accuracy_dist)

# visualize histogram
ablation_lines_np = np.array(ablation_lines) + base_acc
ablation_lines_flatten = ablation_lines_np.flatten()
hist, bin_edges = np.histogram(ablation_lines_flatten, bins=len(accuracy_dist))
plt.figure(figsize = [7, 7])
plt.bar(bin_edges[:-1], hist, width=1, color='blue', alpha=0.05)
plt.xlim(min(bin_edges), max(bin_edges))
plt.xlabel('Accuracy', fontsize=13)
plt.ylabel('# Heads', fontsize=13)
plt.hist(ablation_lines_flatten, bins=len(hist))
plt.title('Task 3.2',fontsize=15)
plt.axvline(x=base_acc, color='r')
plt.show(block=True)

