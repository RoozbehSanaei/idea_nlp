#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:26:02 2017

@author: arlittr
"""

import w2v_word_clusters
import pandas as pd
from tqdm import tqdm
import pre_processing
import similarity_functions
import numpy


#load file
brad_file = '/home/roozbeh/data/wiki/data/n50_de_bc.csv'
ryan_file = '/home/roozbeh/data/wiki/data/n50_de_rma.csv'


bc = pd.read_csv(brad_file)
rma = pd.read_csv(ryan_file)

tqdm.pandas(desc="Make Spacy Tokens")
bc['tokens'] = bc.ix[:,3].progress_apply(lambda x: pre_processing.cleanPassage(x))    
bc['lemmas'] = bc['tokens'].apply(lambda x: pre_processing.getLemmas(x))
sentences = list(bc['lemmas'])
probs_cutoff_lower = pre_processing.findMeaningfulCutoffProbability([t for tok in bc['tokens'] for t in tok])
tqdm.pandas(desc="Remove redundant lemmas")
selected_lemmas = bc['tokens'].progress_apply(lambda x: pre_processing.makeNodelist(x,probs_cutoff_lower))




#cluster based on words
cluster_words = w2v_word_clusters.w2v_word_clusters(selected_lemmas)


M = [[similarity_functions.num_word_similarity(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
M = [[similarity_functions.total_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
#M = [[similarity_functions.max_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
#M = [[similarity_functions.vec_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
clusters =  [M[i].index(max(M[i])) for i in range(len(M))]






#cluster based on sentences
import w2v_sentence_clusters
clusters = w2v_sentence_clusters.w2v_sentence_clusters(sentences)

print(clusters)


#jaccard evaulaton
import jaccard                

data_ryan = [[rma.ix[:,0][i],str(rma.ix[:,1][i])] for i in range(len(clusters))]
data_bc = [[bc.ix[:,1][i],str(bc.ix[:,2][i])] for i in range(len(clusters))]
data_cl = [[str(clusters[i]),str(bc.ix[:,2][i])] for i in range(len(clusters))]
d1 = pre_processing.david_dict(data_ryan)
d2 = pre_processing.david_dict(data_bc)
d3 = pre_processing.david_dict(data_cl)
jaccard.greedy_match_sets(d1, d3)
jaccard.greedy_match_sets(d1, d2)


#save to file
L1 = [[clusters[i],bc['ID'][i]] for i in range(len(selected_lemmas))]
L1.sort(key=lambda x: x[0])
results = pd.DataFrame(L1)
results.to_csv("clusters_w2v_h.csv")
