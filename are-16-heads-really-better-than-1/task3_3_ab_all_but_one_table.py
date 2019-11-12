import numpy as np
import matplotlib.pyplot as plt
import sys

task = sys.argv[1]
filename = 'all_but_one_' + task + '.txt'
f = open(filename, 'r', errors='ignore')
lines = f.readlines()
f.close()

# read accuracy difference into 2D array of the format [layer][head]
acc_diff = lines[-1].split('Layer:')
for i, layer in enumerate(acc_diff):
	acc_diff[i] = layer.split('\t')[1:]
acc_diff = acc_diff[1:]
acc_diff = np.array(acc_diff).astype(np.float)
acc_average_by_layer = np.mean(acc_diff, axis = 1)
acc_max_by_layer = np.max(acc_diff, axis=1)


m, = acc_average_by_layer.shape
acc_average_by_layer = acc_average_by_layer.tolist()
acc_average_by_layer = [[round(item*10000)/100] for item in acc_average_by_layer]
acc_max_by_layer = [[round(item*10000)/100] for item in acc_max_by_layer]
columns = ["Ave Accuracy Difference"]
rows = ["Layer "+str(i+1) for i in range(m)]

fig = plt.figure()
#ax = fig.add_subplot(111)
the_table = plt.table(cellText=acc_average_by_layer,
					  colWidths=[0.5],
                      rowLabels=rows,
                      colLabels=columns,
					  loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1,1.7)
# Removing ticks and spines enables you to get the figure only with table
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
for pos in ['right','top','bottom','left']:
    plt.gca().spines[pos].set_visible(False)
plt.show()
fig.savefig('all_but_one_average_'+task+'.png')

columns = ["Max Accuracy Difference"]

fig = plt.figure()
#ax = fig.add_subplot(111)
the_table = plt.table(cellText=acc_max_by_layer,
					  colWidths=[0.5],
                      rowLabels=rows,
                      colLabels=columns,
					  loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
the_table.scale(1,1.7)
# Removing ticks and spines enables you to get the figure only with table
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
for pos in ['right','top','bottom','left']:
    plt.gca().spines[pos].set_visible(False)
plt.show()
fig.savefig('all_but_one_max_'+task+'.png')
