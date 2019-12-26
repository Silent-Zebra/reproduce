import re
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.font_manager import FontProperties

filepath = 'prune_iwslt_out_all.txt'
f = open('prune_iwslt_out_all.txt', 'r', errors='ignore')
string = f.read()
epoch = [(num) for num in re.findall("Epoch (.*?)\n", string)]
bleus = [float(num) for num in re.findall("BLEU score: 	(.*?)\n", string)]
np_bleus = np.array(bleus)
np_bleus = np.reshape(np_bleus, (-1, 10))
np_bleus_transpose = np.transpose(np_bleus)
#print(np_bleus)
#print(np_bleus_transpose)

labels = [0,10,20,30,40,50,60,70,80,90] # percentage pruned
reference = copy.deepcopy(np_bleus_transpose[0]) # avoid aliasing
for i in range(len(epoch)):
	epoch[i] += '\n' + '['+str(reference[i])+']'
# change all scores to relative decrease score
for i in range(len(np_bleus_transpose)):
	print(reference)
	np_bleus_transpose[i] = np_bleus_transpose[i]/reference*100#(np_bleus_transpose[i]+ 100 - labels[i]-reference)


colors = ['#020770', '#0e4cb0', '#0086fc', '#00cafc', '#00de81', '#7ade00', '#e2e600', '#eb9800', '#e61700', '#590700']
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax = plt.subplot(111)
plt.xlabel('Epoch', fontsize=13)
plt.ylabel("Percentage of un-pruned \n model BLEU score", fontsize=13)
for i, y in enumerate(np_bleus_transpose):
	ax.plot(epoch, y, linestyle='-', color = colors[i], label = str(labels[i]) + '%')
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
plt.show()

#plt.xlabel('Percentage Pruned', fontsize=13)
#plt.ylabel('Accuracy', fontsize=13)
#plt.plot(x, y, color='blue', linestyle='-')
#plt.plot(x, y2, color='green', linestyle='-')
##plt.plot(x, base_acc-np.array(y), color='g', linestyle='--')
#plt.show()



