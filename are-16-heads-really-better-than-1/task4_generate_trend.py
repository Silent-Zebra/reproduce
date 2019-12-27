import re
import matplotlib.pyplot as plt
import numpy as np


def get_prune_accuracy_pair(filepath):
    total_head = 144
    f = open(filepath, 'r')
    base_acc = float(f.readlines()[1].split()[-1])
    f.seek(0)
    prune_accuracy_pair = re.findall(": (.*?)\t(.*?)\n", f.read())
    f.close()
    return prune_accuracy_pair, total_head, base_acc


def get_prune_bleu_pair(filepath):
    f = open(filepath, 'r')
    string = f.read()
    prune_profile = re.findall("Evaluating following profile: \t(.*?)\n", string)
    prune_profile = [line.split() for line in prune_profile]
    total_head = len(prune_profile[-1]) + (len(prune_profile[-1]) - len(prune_profile[-2]))
    bleu_scores = re.findall("BLEU score: 	(.*?)\n", string)
    x = [str(round(len(line)/total_head*100)) for line in prune_profile]
    x = [str(num*10) for num in range(10)]
    y = [float(score) for score in bleu_scores]
    return x, y


def draw_wmt(filepath_acc, filepath_imp):
	# get (percentage pruned, performance) pair
	accx, accy = get_prune_bleu_pair(filepath_acc)
	impx, impy = get_prune_bleu_pair(filepath_imp)

	plt.style.use('seaborn-whitegrid')
	fig, ax = plt.subplots()
	plt.xlabel('Percentage Pruned', fontsize=13)
	plt.ylabel('BLEU', fontsize=13)
	plt.plot(accx, accy, color='green', linestyle='-')
	plt.plot(impx, impy, color='blue', linestyle='-')
	# leg=ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize=12)
	plt.show()


def draw_bert(filepath1, filepath2):
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
	plt.show()


model = 'bert'
if model.lower() == 'bert':
	filepath1 = "MNLIprunesummary.txt"
	filepath2 = "MNLI_prune_acc_summary.txt"
	draw_bert(filepath1, filepath2)
else:
	filepath_acc = 'prune_iwslt_acc_out.txt' # file path to pruning by accuracy output
	filepath_imp = 'prune_iwslt_out_last.txt' # file path to pruning by head importance
	draw_wmt(filepath_acc, filepath_imp)







