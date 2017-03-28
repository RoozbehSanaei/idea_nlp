from __future__ import division
import csv
import copy
import itertools
import tqdm
import numpy as np




def jaccard(s1,s2):
    j = len(s1|s2)/(len(s1)+len(s2)-len(s1&s2))
    return j

def weighted_jaccard(s1,s2):
    j = (len(s1|s2)**2)/(len(s1)+len(s2)-len(s1&s2))

def coverage(s1,s2):
    return len(s1&s2)/len(s1)

#Do a shitty greedy version cause fuck this bs
def greedy_match_sets(r1,r2,f,max_score):

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
    for i, ss_key in tqdm.tqdm(enumerate(ss_keys)):
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
    #pretty-print results
    print("____________________________\n")
    print("Score: {}\n".format(score/max_score))
    #print matchings
    for i, l in enumerate(matchings):
        print("{}:".format(ss_keys[i]))
        for j in l:
            print("\t{}".format(sl_keys[j]))
