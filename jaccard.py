from __future__ import division
import csv
import copy
import itertools
import numpy as np
import random
import statistics
import pre_processing


def cross(v1,v_t):
	M = np.zeros((len(v_t),len(v_t)))
	for V1 in v1:
		for v_1 in V1:
			for v_2 in V1:
					M[v_t.index(v_1),v_t.index(v_2)]=1
	return(M)


def set_of_values(d1):
	v_t = set()
	for v in d1.values():
		v_t = v_t | v
	return(list(v_t))


def co_occurance_score(d1,d2,v_t):
	v1 = list(d1.values())
	v2 = list(d2.values())
	M1 = cross(v1,v_t)
	M2 = cross(v2,v_t)
	return(float(np.linalg.norm(M1-M2)))


def import_human(f):
	with open(f,'r') as f:
		r = csv.reader(f)
		data = list(r)
		data = data[1:]
	#reformat data into a dict with the catagory name, and a set of the UIDs of the contained documents
	cats = [d[0] for d in data if "noise" not in d[0]] #0 uses supersets, 1 uses synthesized
	dout = {c:set([int(d[1]) for d in data if d[0]==c]) for c in cats}
	dout['noise'] = set([int(d[1]) for d in data if 'noise' in d[0]])
	#for c in dout.keys():
	#    print(len(dout[c]))
	return dout


def jaccard(s1,s2):
	j = len(s1|s2)/(len(s1)+len(s2)-len(s1&s2))
	return j

def weighted_jaccard(s1,s2):
	j = (len(s1|s2)**2)/(len(s1)+len(s2)-len(s1&s2))

def coverage(s1,s2):
	return len(s1&s2)/len(s1)

#Do a shitty greedy version cause fuck this bs
def greedy_match_sets(r1,r2):

	#compute the pairwise jaccard score matrix
	if len(r1)<len(r2):
		ss = r1
		sl = r2
	else:
		ss = r2
		sl = r1
	ss_keys = list(ss.keys())
	ss_keys.sort()
	sl_keys = list(sl.keys())
	sl_keys.sort()
	#scores = np.zeros((len(ss),len(sl)))
	scores = []
	for i, ss_key in enumerate(ss_keys):
		scores.append([])
		for j, sl_key in enumerate(sl_keys):
			scores[i].append(len(ss[ss_key])*coverage(ss[ss_key], sl[sl_key]))
	#First, use the excess to find the greatest scores and add those together
	matchings = [[] for _ in range(len(ss_keys))]
	#flatten the scores list
	score_pairs = []
	for i, l in enumerate(scores):
		for j, s in enumerate(l):
			score_pairs.append((i,j,s))
	used = set()
	pairs = list(itertools.product(range(len(ss_keys)),range(len(sl_keys))))
	pairs.sort(key=lambda x:scores[x[0]][x[1]])
	pairs.reverse()
	for p in pairs:
		if p[1] not in used:
			used.add(p[1])
			matchings[p[0]].append(p[1])
	#calculate total score->total number of overlapping items/total number of items
	score = 0
	for i, l in enumerate(matchings):
		for j in l:
			score+=len(ss[ss_keys[i]]&sl[sl_keys[j]])
	return float(score)



def save_histogram(f,n_cats,match_type):
	d1 = import_human('data/'+f+'.csv')
	v_t = set_of_values(d1)
	#prep work to generate random computer sets
	ids = set()
	for c in d1.keys():
		ids = ids | d1[c]
	ids = list(ids) 
	ids.sort()

	def uniform_random_alloc(n, ids):
		while True:
			inds = [random.randint(0,n-1) for _ in ids]
			bins = {i:set() for i in range(0,n)}
			for i, id_ in enumerate(ids):
				bins[inds[i]].add(id_)
			if all([len(bins[b])>0 for b in bins.keys()]):
				break
		return bins



	#Run simulation
	scores = []
	for _ in range(500):
		d_temp = uniform_random_alloc(n_cats, ids)
		
		if match_type=='greedy_match_sets':		
			score = greedy_match_sets(d1, d_temp)/len(ids)
		
		if match_type=='co_occurance_score':
			score = co_occurance_score(d1, d_temp,v_t)		
		
		scores.append(score)
	
	mean = statistics.mean(scores)
	std = statistics.stdev(scores)
	print("Average random score: {}, Random std. dev: {}, Mean+3 std.: {}".format(mean, std, mean+3*std))
	import matplotlib.pyplot as plt
	plt.hist(scores)
	plt.axvline(np.array(scores).mean(), color='r', linestyle='dashed', linewidth=2)
	filename = f + "(n_clusters :" + str(n_cats)+") -" + match_type
	plt.title(filename)
	import pylab
	pylab.savefig("results/"+filename)
	print(filename)
	plt.close()
	#plt.show()

