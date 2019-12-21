import re
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats

# read file
f = open("MNLIOut.txt", 'r', errors='ignore')
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
print("Number of examples:", num_example)
#print("batch size:", batch_size)
#print("speeds:", evaluation_speed)
#print("base accuracy:", base_acc)
#print("ablation line:", ablation_lines)

# convert from str to float
for i in range(len(ablation_lines)):
	for j in range(len(ablation_lines[i])):
		ablation_lines[i][j] = float(ablation_lines[i][j])

# print(ablation_lines)
np.savetxt("32BERT.csv", ablation_lines, delimiter=",")

ablation_lines = np.array(ablation_lines)

flattened_differences = ablation_lines.flatten()


num_test_datapoints = num_example

print("Mean diff = {:.3f}".format(flattened_differences.mean()))

# Test significance
for i in range(len(flattened_differences)):
	p = stats.binom_test((flattened_differences[i] + base_acc) * num_test_datapoints,
						 num_test_datapoints, base_acc, alternative='two-sided')
	if p < 0.01:
		print("p = {:.3f}".format(p))
		print("Index = {}".format(i))
		print("Value = {}".format(flattened_differences[i]))

# build histogram from ablation accuracy
accuracy_dist = {}
for layer in ablation_lines:
	for head in layer:
		key = math.floor((base_acc+ head)*10000)/10000
		accuracy_dist[key] = accuracy_dist.get(key, 0) + 1
#print(accuracy_dist)

print(base_acc)

# visualize histogram
ablation_lines += base_acc
ablation_lines_flatten = ablation_lines.flatten()
hist, bin_edges = np.histogram(ablation_lines_flatten, bins=len(accuracy_dist))
plt.figure(figsize = [7, 7])
plt.bar(bin_edges[:-1], hist, width=1, color='white')
plt.xlim(min(bin_edges), max(bin_edges))
plt.xlabel('Accuracy', fontsize=13)
plt.ylabel('# Heads', fontsize=13)
plt.hist(ablation_lines_flatten, bins=len(hist))
plt.title('Task 3.2',fontsize=15)
plt.axvline(x=base_acc, color='r')
plt.show(block=True)

