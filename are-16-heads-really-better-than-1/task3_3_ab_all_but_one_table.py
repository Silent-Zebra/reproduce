import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_table(acc, task):
    m, = acc.shape
    acc = acc.tolist()
    acc = [["{:.2f}%".format(round(item * 10000) / 100)] for
                            item in acc]
    columns = ["Accuracy Difference"]
    rows = ["Layer " + str(i + 1) for i in range(m)]

    fig = plt.figure()
    # ax = fig.add_subplot(111)
    the_table = plt.table(cellText=acc,
                          colWidths=[0.4],
                          rowLabels=rows,
                          colLabels=columns,
                          loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1.7)
    # Removing ticks and spines enables you to get the figure only with table
    plt.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False,
                    labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)
    plt.show()
    fig.savefig('all_but_one_' + task + '.png')


# task = sys.argv[1]
task="MNLI"
filename = 'all_but_one_' + task + '.txt'
f = open(filename, 'r', errors='ignore')
lines = f.readlines()
f.close()

base_acc = float(lines[-3].split()[-1])
num_example = None
for line in lines:
	if 'Num examples' in line:
		num_example = float(line.split()[-1])

# read accuracy difference into 2D array of the format [layer][head]
acc_diff = lines[-1].split('Layer:')
for i, layer in enumerate(acc_diff):
	acc_diff[i] = layer.split('\t')[1:]
acc_diff = acc_diff[1:]
acc_diff = np.array(acc_diff).astype(np.float)
acc_average_by_layer = np.mean(acc_diff, axis = 1)
acc_best_by_layer = np.max(acc_diff, axis = 1)

# print(base_acc)

# Statistical significance testing
# All values
# acc_diff_flat = acc_diff.flatten()
# for i in range(len(acc_diff_flat)):
# 	p = stats.binom_test((acc_diff_flat[i] + base_acc) * num_example,
#                          num_example, base_acc, alternative='two-sided')
# 	if p < 0.01:
# 		print("p = {:.3f}".format(p))
# 		print("Index = {}".format(i))
# 		print("Value = {}".format(acc_diff_flat[i]))

# Min values
for i in range(len(acc_best_by_layer)):
	p = stats.binom_test((acc_best_by_layer[i] + base_acc) * num_example,
                         num_example, base_acc, alternative='two-sided')
	if p < 0.01:
		print("p = {:.3f}".format(p))
		print("Index = {}".format(i))
		print("Value = {}".format(acc_best_by_layer[i]))

plot_table(acc_average_by_layer, task)
plot_table(acc_best_by_layer, task)

print(acc_best_by_layer.flatten())
