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
	
	bad_chars = "".join(string.punctuation)
	for c in bad_chars: rawtext = rawtext.replace(c, "")
	
	tokens = parser(rawtext)

	tokens = [tok for tok in tokens if tok not in stopwords.words('english')]
	
	return tokens



def relevance(w):
	score = model.similarity('vehicle',w)+model.similarity('city',w)
	return score

def makelist(rawtext,limitPOS=None):
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
	
		relevant = False
		if inWiki:
			if relevance(tok.lemma_)>0.2:
				relevant = True

		if goodPOS and notStopword and notSymbol and inWiki and relevant :
			nodes.append((tok.lemma_,tok.pos_))
	return nodes  


def similarity(l1,l2):
	r = 0
	if ((l1[0] in model.wv.vocab) & (l2[0] in model.wv.vocab)): 
		r = model.similarity(l1[0],l2[0])
	return(r)



def overall_sim(lemmas1,lemmas2):
	s = 0

	if (lemmas1 != []) & (lemmas2 != []):
		s = max([max([model.similarity(l1[0],l2[0]) for l1 in lemmas1]) for l2 in lemmas2])

	return s


def sort_index(X):
	return sorted(range(len(X)), key=lambda k: -X[k])


sentences_strs = file_utils.extract_col('/home/roozbeh/data/wiki/data/n1400_dn.csv',2)
human_strs = file_utils.extract_col('/home/roozbeh/data/wiki/data/n50_de_bc.csv',2)

print("generate bag of lemmas:")
bag_of_lemmas = [list(set(makelist(x))) for x in tqdm(sentences_strs)]
bag_of_lemmas_h = [list(set(makelist(x))) for x in human_strs]


all_lemmas = []
for lemmas_ in bag_of_lemmas: 
		all_lemmas = all_lemmas+lemmas_

lemma_count = [(t[0],t[1],all_lemmas.count(t),relevance(t[0])) for t in set(all_lemmas) if t[0] in model.wv.vocab]
lemma_count.sort(key=lambda x: x[1],reverse=True)

file_utils.rows_to_csv(lemma_count,['token','POS','count','relevance'],'tokens.csv')

"""
M = [[overall_sim(lemma1,lemma2) for lemma1 in bag_of_lemmas_h] for lemma2 in tqdm(bag_of_lemmas_h)]
strs = [[human_strs[i],human_strs[sort_index(M[i])[0]],human_strs[sort_index(M[i])[1]],bag_of_lemmas_h[sort_index(M[i])[0]],bag_of_lemmas_h[sort_index(M[i])[1]]] for i in range(len(M))]
file_utils.rows_to_csv(strs,['S','S1','S2','L1','L2'],'sents.csv')
"""

print("generate similarity:")
SUM = numpy.sum(X,axis=0)

X_pa
sorted_lemmas = [[]]*len(X)
for i in tqdm(range(len(X))):
	s = sort_index(X[i])
	sorted_lemmas[i] = [(lemma_count[s[j]],X[i][s[j]]) for j in range(len(X))]
		


X_path = numpy.zeros((len(lemma_count),len(lemma_count)))
X_w2v = numpy.zeros((len(lemma_count),len(lemma_count)))
X_wup = numpy.zeros((len(lemma_count),len(lemma_count)))

from nltk.corpus import wordnet as wn

for i1 in tqdm(range(len(lemma_count))):
	for i2 in range(len(lemma_count)):
		l1 = lemma_count[i1][0]
		l2 = lemma_count[i2][0]
		s1 = wn.synsets(l1)
		s2 = wn.synsets(l2)
		if (s1!=[]) & (s2!=[]):
			X_path[i1][i2] = wn.path_similarity(s1[0],s2[0])
			X_wup[i1][i2] = wn.wup_similarity(s1[0],s2[0])
			X_w2v[i1][i2] = model.similarity(l1,l2)




print(dog.path_similarity(cat))
print(dog.lch_similarity(cat))
print(dog.wup_similarity(cat))