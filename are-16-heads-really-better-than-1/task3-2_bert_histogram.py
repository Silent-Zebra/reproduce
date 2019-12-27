import re
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats

# read file
task = 'SST2'
# change this filename if you output to different filename
filename = task+"out.txt"
f = open(filename, 'r', errors='ignore')
lines = f.readlines()
f.close()

# prepare data from file
# data include total number of examples,
# evaluation speed, and head ablation score
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

# convert from str to float
for i in range(len(ablation_lines)):
	for j in range(len(ablation_lines[i])):
		ablation_lines[i][j] = float(ablation_lines[i][j])

np.savetxt("32_"+task+"_BERT.csv", ablation_lines, delimiter=",")

ablation_lines = np.array(ablation_lines)

flattened_differences = ablation_lines.flatten()

num_test_datapoints = num_example

print('Task', task)
print("Mean diff = {:.3f}".format(flattened_differences.mean()))

# Test significance
for i in range(len(flattened_differences)):
	p = stats.binom_test((flattened_differences[i] + base_acc) * num_test_datapoints,
						 num_test_datapoints, base_acc, alternative='two-sided')
	# print index of values that has statistical significance.
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
print(base_acc)

# visualize histogram
ablation_lines += base_acc
ablation_lines_flatten = ablation_lines.flatten()
hist, bin_edges = np.histogram(ablation_lines_flatten, bins=len(accuracy_dist))
plt.figure(figsize = [8, 8])
plt.bar(bin_edges[:-1], hist, width=1, color='white')
plt.xlim(min(bin_edges), max(bin_edges))
plt.xlabel('Accuracy', fontsize=20)
plt.ylabel('# Heads', fontsize=20)
plt.hist(ablation_lines_flatten, bins=len(hist))
plt.axvline(x=base_acc, color='r')
plt.tick_params(labelsize=17)
plt.show(block=True)


