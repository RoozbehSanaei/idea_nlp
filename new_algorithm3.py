#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:26:02 2017

@author: arlittr
"""

from spacy.en import English
import pandas as pd
from nltk.corpus import stopwords as stopwords
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import numpy
import gensim
global model 
import file_utils

model = gensim.models.Word2Vec.load("wiki_files/wiki.en.word2vec.model")    
parser = English()


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



def relevance(w):
	score = model.similarity('satellite',w)+model.similarity('technology',w)
	return score


def makelist(rawtext,limitPOS=None):
	generic = []
	tokens = cleanPassage(rawtext)
	if limitPOS:
		GOODPOS = limitPOS
	else:
		GOODPOS = ['NOUN','PROPN','VERB','ADJ','ADV']
	SYMBOLS = " ".join(string.punctuation).split(" ")#+[">","<","|","/","\"]
	nodes = []
	lemmas = []
	for tok in tokens:
		goodPOS = tok.pos_ in GOODPOS 
		notStopword = tok.orth_ not in stopwords.words('english')
		notSymbol = tok.orth_ not in SYMBOLS
		inWiki = tok.lemma_ in model.wv.vocab
		notGeneric = tok.lemma_ not in generic

		relevant = False
		if inWiki:
			if relevance(tok.lemma_)>0.4:
				relevant = True

		if goodPOS and notStopword and notSymbol and inWiki and relevant and notGeneric:
			nodes.append(tok.lemma_)
	return nodes  

def sort_index(X):
	return sorted(range(len(X)), key=lambda k: -X[k])

f = 'satellite_news'
sentences_strs = file_utils.extract_col('/home/roozbeh/data/wiki/data/'+f+'.csv',2)

bag_of_lemmas = [list(set(makelist(x))) for x in tqdm(sentences_strs)]

limit_sets = [([],0)]
filter_sets = [[]]

all_tokens = []
for tokens_ in bag_of_lemmas: 
		all_tokens = all_tokens+tokens_
token_count = [ (i,all_tokens.count(i)) for i in set(all_tokens) ]
token_count.sort(key=lambda x: x[1],reverse=True)



X = [[model.similarity(w1[0],w2[0]) for w1 in token_count] for w2 in token_count]
sorted_tokens = [[]]*len(X)
for i in tqdm(range(len(X))):
	s = sort_index(X[i])
	sorted_tokens[i] = [(token_count[s[j]],X[i][s[j]]) for j in range(len(X))]


n = 20;
s = 1+n+n**2
m=5
from tqdm import trange
j = 0
while (j<min(s,len(limit_sets))):
	limit_set = limit_sets[j][0]
	filter_set = filter_sets[j]


	all_tokens = []
	for tokens_ in bag_of_lemmas: 
		if all(t in tokens_ for t in limit_set):
			all_tokens = all_tokens+tokens_

	filtered_tokens = [x for x in all_tokens if x not in filter_set]


	token_count = [ (i,filtered_tokens.count(i)) for i in set(filtered_tokens) ]
	token_count.sort(key=lambda x: x[1],reverse=True)
	
	top = token_count[0:n]
	#top = [t for t in token_count if t[1]>m]



	filter_set = filter_set+[x[0] for x in top]

	#limit_sets = limit_sets + [limit_set]
	for i in range(len(top)):
		limit_sets = limit_sets + [(limit_set + [top[i][0]],top[i][1])]
	
	for i in range(len(top)):
		filter_sets = filter_sets + [filter_set]
	j = j + 1

"""

results=[]
#for i in range(s,len(limit_sets)):
for i in range(n+1,s):
	results.append(limit_sets[i])

"""

results = limit_sets

counter = 0
sents = [] 
for result in tqdm(results):
	counter = counter + 1
	limit = result[0]
	n = result[1]
	for i in range(len(bag_of_lemmas)): 
		if all(t in bag_of_lemmas[i] for t in limit):
			sents.append((counter,n,limit,sentences_strs[i]))

import file_utils
file_utils.rows_to_csv(sents,['cluster','n','keywords','idea'],'res-'+f+'.csv')


#create the mind-map file


Keywords = [];
L = ['id,value','ideas,']; 
for j in range(0,len(limit_sets)):	
	if (len(limit_sets[j][0])==2):
		if (limit_sets[j][0][0]!=limit_sets[j-1][0][0]):
			L.append('ideas.'+limit_sets[j][0][0]+',');
			Keywords.append(limit_sets[j][0][0]);
		L.append('ideas.'+limit_sets[j][0][0]+'.'+limit_sets[j][0][1]+',');
		


file_utils.save_to_txt_file(L,'map.csv');


