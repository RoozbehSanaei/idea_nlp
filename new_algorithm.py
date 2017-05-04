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


import gensim
global model 
model = gensim.models.Word2Vec.load("wiki_files/wiki.en.word2vec.model")    

def relevance(w):
	score = model.similarity('vehicle',w)+model.similarity('city',w)
	return score


jj = 0
inputbasepath = '/home/roozbeh/data/wiki/data/'
outputbasepath = '/home/roozbeh/data/wiki/data/'
#basename = 'Distributed Experience and Novice (superset) clean TEST SAMPLE'
basename = 'n1400_dn'
fileextension = '.csv'
path = inputbasepath + basename + fileextension

print("loading file...",end="",flush=True)
gn = pd.read_csv(path)
print("done!",flush=True)
	
tqdm.pandas(desc="Make Spacy Tokens")
gn['tokens'] = gn.ix[:,2].progress_apply(lambda x: cleanPassage(x))    
gn['lemmas'] = gn['tokens'].apply(lambda x: getLemmas(x))
sentences_strs = [" ".join(sent) for sent in gn['lemmas'] ]

probs_cutoff_lower = findMeaningfulCutoffProbability([t for tok in gn['tokens'] for t in tok])

tqdm.pandas(desc="Remove redundant tokens")
selected_tokens = gn['tokens'].progress_apply(lambda x: makeNodelist(x))
sentence_tokens = [list(set(tokens)) for tokens in selected_tokens]


all_tokens = []
for tokens_ in sentence_tokens: 
		all_tokens = all_tokens+tokens_

token_count = [(t[0],t[1],all_tokens.count(t),relevance(t[0])) for t in set(all_tokens) if t[0] in model.wv.vocab]
token_count.sort(key=lambda x: x[1],reverse=True)

import file_utils
file_utils.rows_to_csv(token_count,['token','POS','count','relevance'],'tokens.csv')

noun_list = [x for x  in token_count if x[1]=='NOUN']
noun_list.sort(key=lambda x: -x[3])
