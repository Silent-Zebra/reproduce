import re
import matplotlib.pyplot as plt


def get_prune_bleu_pair(filepath):
	total_head = 48
	f = open(filepath, 'r')
	string = f.read()
	prune_profile = re.findall("Evaluating following profile: \t(.*?)\n", string)
	prune_profile = [line.split() for line in prune_profile]
	total_head = len(prune_profile[-1]) + (len(prune_profile[-1]) - len(prune_profile[-2]))
	bleu_scores = re.findall("BLEU score: 	(.*?)\n", string)
	x = [str(num*10) for num in range(10)]
	y = [float(score) for score in bleu_scores]
	return x, y

# output of pruning only on enc-enc layer
en_file = 'prune_iwslt_enc_only_lastcp.txt'
# output of pruning only on enc-dec layer
ende_file = 'prune_iwslt_encdec_only_lastcp.txt'
# output of pruning only on dec-dec layer
de_file = 'prune_iwslt_dec_only_lastcp.txt'

# get (percentage pruned, bleu) pair
enx, eny = get_prune_bleu_pair(en_file)
endex, endey = get_prune_bleu_pair(ende_file)
dex, dey = get_prune_bleu_pair(de_file)

# draw plot
plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots()
plt.xlabel('Percentage Pruned', fontsize=13)
plt.ylabel('BLEU', fontsize=13)
plt.plot(enx, eny, color='red', linestyle='-', label='Enc-Enc')
plt.plot(endex, endey, color='blue', linestyle='-', label = 'Enc-Dec')
plt.plot(dex, dey, color='green', linestyle='-', label = 'Dec-Dec')
leg=ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, fontsize=12)
plt.show()

