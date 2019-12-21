import re
import matplotlib.pyplot as plt
import numpy as np

total_head = 144
f = open("MNLIprunesummary.txt", 'r')
base_acc = float(f.readlines()[1].split()[-1])
f.seek(0)
f = open("MNLIprunesummary.txt", 'r')
prune_accuracy_pair = re.findall(": (.*?)\t(.*?)\n", f.read())
f.close()
x = [float(pair[0])/total_head*100 for pair in prune_accuracy_pair]
y = [float(pair[1]) for pair in prune_accuracy_pair]
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.axes()
plt.xlabel('Precentage Pruned', fontsize=13)
plt.ylim(0)
plt.ylabel('Accuracy', fontsize=13)
plt.plot(x, y, color='blue', linestyle='-')
#plt.plot(x, base_acc-np.array(y), color='g', linestyle='--')
plt.show()



