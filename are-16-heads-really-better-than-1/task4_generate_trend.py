import re
import matplotlib.pyplot as plt
import numpy as np


def get_prune_accuracy_pair(filepath):
    # NOTE change (dynamic?) for WMT
    total_head = 144
    f = open(filepath, 'r')
    base_acc = float(f.readlines()[1].split()[-1])
    f.seek(0)
    prune_accuracy_pair = re.findall(": (.*?)\t(.*?)\n", f.read())
    f.close()
    return prune_accuracy_pair, total_head, base_acc

filepath1 = "MNLIprunesummary.txt"
filepath2 = "MNLI_prune_acc_summary.txt"

prune_accuracy_pair1, total_head, base_acc = get_prune_accuracy_pair(filepath1)
prune_accuracy_pair2, total_head2, base_acc2 = get_prune_accuracy_pair(filepath2)

x = [float(pair[0])/total_head*100 for pair in prune_accuracy_pair1]
y = [float(pair[1]) for pair in prune_accuracy_pair1]
y2 = [float(pair[1]) for pair in prune_accuracy_pair2]
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
plt.xlabel('Percentage Pruned', fontsize=13)
plt.ylim(0)
plt.ylabel('Accuracy', fontsize=13)
plt.plot(x, y, color='blue', linestyle='-')
plt.plot(x, y2, color='green', linestyle='-')
#plt.plot(x, base_acc-np.array(y), color='g', linestyle='--')
plt.show()



