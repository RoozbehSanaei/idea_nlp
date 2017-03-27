#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:26:02 2017

@author: arlittr
"""

from spacy.en import English
parser = English()

import pandas as pd
from nltk.corpus import stopwords as stopwords
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import numpy


def cleanPassage(rawtext):
	#some code from https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
	
	#if data is bad, return empty
	if type(rawtext) is not str:
		return ''
	
	#split text with punctuation
	bad_chars = "".join(string.punctuation)
	for c in bad_chars: rawtext = rawtext.replace(c, "")
	
	#parse 
	tokens = parser(rawtext)

	# stoplist the tokens
	tokens = [tok for tok in tokens if tok not in stopwords.words('english')]
	
	return tokens

def getLemmas(tokens):
	# lemmatize
	lemmas = [tok.lemma_.lower().strip() for tok in tokens]
	return lemmas


def makeNodelist(tokens,limitPOS=None):
#    BADPOS = ['PUNCT','NUM','X','SPACE']
	if limitPOS:
		GOODPOS = limitPOS
	else:
		GOODPOS = ['NOUN','PROPN','VERB','ADJ','ADV']
	SYMBOLS = " ".join(string.punctuation).split(" ")#+[">","<","|","/","\"]
	probs_cutoff_upper = -7.6 #by inspection of sample data
	nodes = []
	lemmas = []
	for tok in tokens:
		goodPOS = tok.pos_ in GOODPOS 
		notStopword = tok.orth_ not in stopwords.words('english')
		notSymbol = tok.orth_ not in SYMBOLS
		isMeaningful = tok.prob > probs_cutoff_lower and tok.prob < probs_cutoff_upper
		
		if goodPOS and notStopword and notSymbol and isMeaningful:
			nodes.append((tok.lemma_,tok.pos_))
			lemmas.append(tok.lemma_)
	return nodes  

def findMeaningfulCutoffProbability(alltokens):
	probs = [tok.prob for tok in alltokens]
	#set probs_cutoff by inspection by looking for the elbow on the plot of sorted log probabilities
#    probs_cutoff = 500
#    probs_cutoff = probs[int(input("By inspection, at which rank is the elbow for the log probability plot? [integer]"))]
	
	#removing the lowest observed probability seems to remove most of the spelling errors
	probs_cutoff_lower = min(probs)
	return probs_cutoff_lower


jj = 0
files = ['Distributed Experience and Novice (superset) clean','n1400 distributed novice','n1400 group experienced','n100 group novice','n100 distributed experienced']
inputbasepath = '/home/roozbeh/data/wiki/'
outputbasepath = '/home/roozbeh/data/wiki/'
#basename = 'Distributed Experience and Novice (superset) clean TEST SAMPLE'
basename = files[jj]
fileextension = '.csv'
path = inputbasepath + basename + fileextension

print("loading file...",end="",flush=True)
gn = pd.read_csv(path)
print("done!",flush=True)
	
tqdm.pandas(desc="Make Spacy Tokens")
gn['tokens'] = gn.ix[:,1].progress_apply(lambda x: cleanPassage(x))    
gn['lemmas'] = gn['tokens'].apply(lambda x: getLemmas(x))

probs_cutoff_lower = findMeaningfulCutoffProbability([t for tok in gn['tokens'] for t in tok])

tqdm.pandas(desc="Remove redundant tokens")
selected_tokens = gn['tokens'].progress_apply(lambda x: makeNodelist(x))
sentence_tokens = [list(set(tokens)) for tokens in selected_tokens]


limit_sets = [[]]
filter_sets = [[]]

n = 20;
from tqdm import trange
for j in trange(421):
	#print(limit_set)
	limit_set = limit_sets[j]
	filter_set = filter_sets[j]


	all_tokens = []
	for tokens_ in sentence_tokens: 
		if all(t in tokens_ for t in limit_set):
			all_tokens = all_tokens+tokens_

	filtered_tokens = [x for x in all_tokens if x not in filter_set]


	token_count = [ (i,filtered_tokens.count(i)) for i in set(filtered_tokens) ]
	token_count.sort(key=lambda x: x[1],reverse=True)

	top = token_count[0:n]
	filter_set = filter_set+[x[0] for x in top]

	#limit_sets = limit_sets + [limit_set]
	for i in range(len(top)):
		limit_sets = limit_sets + [limit_set + [top[i][0]]]
	
	for i in range(len(top)):
		filter_sets = filter_sets + [filter_set]


results=[]
for i in range(421,len(limit_sets)):
	results.append([int((i-421)/20)]+limit_sets[i])

import pandas as pd
panda_results = pd.DataFrame(results)
panda_results.to_csv("new_clusters.csv")

