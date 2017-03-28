from __future__ import division
import csv
import copy
import itertools
import tqdm
import numpy as np
import random
import statistics

def import_human():
    with open('samples.csv','r') as f:
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

def import_computer():
    with open('clusters.csv','r') as f:
        r = csv.reader(f)
        data = list(r)
    #reformat data into a dict with the catagory name, and a set of the UIDs of the contained documents
    cats = [d[0] for d in data if "noise" not in d[0]]
    dout = {c:set([int(d[1]) for d in data if d[0]==c]) for c in cats}
    #print(dout)
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
    max_score = 100
    #pretty-print results
    #print("____________________________\n")
    #print matchings

    ##for i, l in enumerate(matchings):
    ##    print("{}:".format(ss_keys[i]))
    ##    for j in l:
    ##        print("\t{}".format(sl_keys[j]))

    #print("Score: {}\n".format(score/max_score))
    #print("Number of human sets: {}\n".format(len(r1)))
    #print("Number of computer sets: {}\n".format(len(r2)))
    return score/max_score, matchings

def run_all():
    d1 = import_human()
    d2 = import_computer()
    score, matchings = greedy_match_sets(d1, d2)
    print(len(d1),len(d2))
    print("Total score: {}".format(score))
    #prep work to generate random computer sets
    ids = set()
    for c in d1.keys():
        ids = ids | d1[c]
    ids = list(ids)
    ids.sort()
    #print(len(ids))
    n_cats = len(d2.keys())

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
        score, matchings = greedy_match_sets(d1, d_temp)
        scores.append(score)
    mean = statistics.mean(scores)
    std = statistics.stdev(scores)
    print("Average random score: {}, Random std. dev: {}, Mean+3 std.: {}".format(mean, std, mean+3*std))

if __name__ == '__main__':
    #d1 = import_n50_group_experienced_bc()
    #d2 = import_n50_group_experienced_rma()
    #print(len(d1), len(d2))
    #for c in d.keys():
    #    print c, d[c]
    run_all()
