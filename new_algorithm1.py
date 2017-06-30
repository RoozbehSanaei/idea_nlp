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
	score = model.similarity('vehicle',w)+model.similarity('city',w)
	return score


def makelist(rawtext,limitPOS=None):
	generic = ['city','car','transportation','area','traffic','vehicle','town','transport','system','its','passenger']
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
			nodes.append((tok.lemma_,tok.pos_))
	return nodes  


sentences_strs = file_utils.extract_col('/home/roozbeh/data/wiki/data/n1400_dn.csv',2)
human_strs = file_utils.extract_col('/home/roozbeh/data/wiki/data/n50_de_bc.csv',2)

bag_of_lemmas = [list(set(makelist(x))) for x in tqdm(sentences_strs)]
bag_of_lemmas_h = [list(set(makelist(x))) for x in tqdm(human_strs)]

limit_sets = [[]]
filter_sets = [[]]

n = 20;
s = 1+n+n**2
from tqdm import trange
for j in trange(s):
	limit_set = limit_sets[j]
	filter_set = filter_sets[j]


	all_tokens = []
	for tokens_ in bag_of_lemmas: 
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
#for i in range(s,len(limit_sets)):
for i in range(n+1,s):
	results.append(limit_sets[i])



counter = 0
sents = [] 
for result in tqdm(results):
	counter = counter + 1
	limit = result
	for i in range(len(bag_of_lemmas)): 
		if all(t in bag_of_lemmas[i] for t in limit):
			sents.append((counter,limit,sentences_strs[i]))

import file_utils
file_utils.rows_to_csv(sents,['cluster','keywords','idea'],'clusters2.csv')