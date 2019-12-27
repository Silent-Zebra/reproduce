import re
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import pearsonr

model = 'BERT'

def draw_bert():
	taskx = 'MNLI' # task displayed on x axis
	tasky = 'mnli_mis_' # task displayed on y axis

	# prepare data for x axis
	filenamex = taskx+"out.txt"
	f = open(filenamex, 'r', errors='ignore')
	lines = f.readlines()
	f.close()

	taskx = 'MNLI matched'
	base_accx = float(lines[-3].split()[-1])
	ablation_linesx = lines[-1].split('Layer:')
	for i, layer in enumerate(ablation_linesx):
		ablation_linesx[i] = layer.split('\t')[1:]
	ablation_linesx = ablation_linesx[1:]
	# convert from str to float
	for i in range(len(ablation_linesx)):
		for j in range(len(ablation_linesx[i])):
			ablation_linesx[i][j] = float(ablation_linesx[i][j])

	ablation_linesx = np.array(ablation_linesx).flatten() + base_accx
	print(ablation_linesx)

	# prepare data for y axis
	filenamey = tasky+"out.txt"
	f = open(filenamey, 'r', errors='ignore')
	lines = f.readlines()
	f.close()
	tasky = 'MNLI-Mismatched'
	base_accy = float(lines[-3].split()[-1])
	ablation_linesy = lines[-1].split('Layer:')
	for i, layer in enumerate(ablation_linesy):
		ablation_linesy[i] = layer.split('\t')[1:]
	ablation_linesy = ablation_linesy[1:]
	# convert from str to float
	for i in range(len(ablation_linesy)):
		for j in range(len(ablation_linesy[i])):
			ablation_linesy[i][j] = float(ablation_linesy[i][j])

	ablation_linesy = np.array(ablation_linesy).flatten() + base_accy
	print(ablation_linesy)

	r_score, _ = pearsonr(ablation_linesx, ablation_linesy)
	print("r_score;", r_score)

	fig, ax = plt.subplots()
	plt.scatter(ablation_linesx, ablation_linesy, marker = '+', label='Head ablation accuracies (Pearson r = '+ str(round(r_score*100)/100)+')')
	plt.plot([base_accx], [base_accy], marker='*', markersize=13, color="red", label='Original accuracies')
	leg = ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize=12)
	plt.xlabel(taskx, fontsize=20)
	plt.ylabel(tasky, fontsize=20)
	plt.tick_params(labelsize=17)
	plt.show()

def draw_wmt():
	filenamex = 'wmt_en-fr_out2.txt'
	taskx = 'BLEU on newstest2014'
	filenamey = 'wmt_MTNT_out.txt'
	tasky = 'BLEU on MTNT'

	# open file and read task x
	f = open(filenamex, 'r', errors='ignore')
	lines = f.readlines()
	f.close()

	# read base accuracy and head ablation score
	baseaccx = float(lines[0])
	enen = lines[2:8]
	ende = lines[9:15]
	dede = lines[16:22]

	for i, layer in enumerate(enen):
		layer = layer.rstrip('\n')
		enen[i] = layer.split('\t')[1:]
	for i, layer in enumerate(ende):
		ende[i] = layer.split('\t')[1:]
	for i, layer in enumerate(dede):
		dede[i] = layer.split('\t')[1:]

	# obtain accuracy
	enen = np.array(enen).astype(float) + baseaccx
	ende = np.array(ende).astype(float) + baseaccx
	dede = np.array(dede).astype(float) + baseaccx

	ablation_linesx = list(enen.flatten()) + list(ende.flatten()) + list(dede.flatten())

	# open file and read task y
	f = open(filenamey, 'r', errors='ignore')
	lines = f.readlines()
	f.close()
	baseaccy = float(lines[0])
	enen = lines[2:8]
	ende = lines[9:15]
	dede = lines[16:22]

	for i, layer in enumerate(enen):
		layer = layer.rstrip('\n')
		enen[i] = layer.split('\t')[1:]
	for i, layer in enumerate(ende):
		ende[i] = layer.split('\t')[1:]
	for i, layer in enumerate(dede):
		dede[i] = layer.split('\t')[1:]

	enen = np.array(enen).astype(float) + baseaccy
	ende = np.array(ende).astype(float) + baseaccy
	dede = np.array(dede).astype(float) + baseaccy

	ablation_linesy = list(enen.flatten()) + list(ende.flatten()) + list(dede.flatten())

	r_score, _ = pearsonr(ablation_linesx, ablation_linesy)
	print("r_score;", r_score)

	fig, ax = plt.subplots()
	plt.scatter(ablation_linesx, ablation_linesy, marker = '+', label='Head ablation BLEU (Pearson r = '+ str(round(r_score*100)/100)+')')
	plt.plot([baseaccx], [baseaccy], marker='*', markersize=16, color="red", label='Original BLEU')
	leg=ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize=12)
	plt.xlabel(taskx, fontsize=20)
	plt.ylabel(tasky, fontsize=20)
	plt.tick_params(labelsize=17)
	plt.show()


if model.lower() == 'wmt':
	draw_bert()
else:
	draw_wmt()










