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
            nodes.append(tok.lemma_+' '+tok.pos_)
            lemmas.append(tok.lemma_)
    return lemmas  

def findMeaningfulCutoffProbability(alltokens):
    probs = [tok.prob for tok in alltokens]
    #set probs_cutoff by inspection by looking for the elbow on the plot of sorted log probabilities
#    probs_cutoff = 500
#    probs_cutoff = probs[int(input("By inspection, at which rank is the elbow for the log probability plot? [integer]"))]
    
    #removing the lowest observed probability seems to remove most of the spelling errors
    probs_cutoff_lower = min(probs)
    return probs_cutoff_lower


jj = 0
files = ['n1400 distributed novice','n1400 group experienced','n100 group novice','n100 distributed experienced']
inputbasepath = '/home/roozbeh/data/wiki/'
outputbasepath = '/home/roozbeh/data/wiki/'
#basename = 'Distributed Experience and Novice (superset) clean TEST SAMPLE'
basename = files[jj]
fileextension = '.csv'
path = inputbasepath + basename + fileextension


gn = pd.read_csv(path)
    
tqdm.pandas(desc="Make Spacy Tokens")
gn['tokens'] = gn.ix[:,1].progress_apply(lambda x: cleanPassage(x))    
gn['lemmas'] = gn['tokens'].apply(lambda x: getLemmas(x))

probs_cutoff_lower = findMeaningfulCutoffProbability([t for tok in gn['tokens'] for t in tok])

tqdm.pandas(desc="Remove redundant lemmas")
selected_lemmas = gn['tokens'].progress_apply(lambda x: makeNodelist(x))

words = []
for lemmas in selected_lemmas:
	words = words+lemmas


import gensim
from gensim import corpora, models
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")    

# remove repetitive words or those that are not found in wikipedia vocabulary
words = list(set(words))
words = [w for w in words if w in model.wv.vocab]

tqdm(desc="Make Similarity Matrix")
#make the similarity model
similarity_matrix = [[model.similarity(w1, w2) for w1 in words] for w2 in tqdm(words)]


tqdm(desc="Remove Negative Correlations")
#ignore negative corrolation
import numpy
from tqdm import trange
for i in trange(len(similarity_matrix)):
	for j in range(len(similarity_matrix)):
		if (similarity_matrix[i][j]<0):
			similarity_matrix[i][j]=0

#cluster
import numpy
from sklearn.cluster import AffinityPropagation
af = AffinityPropagation().fit(similarity_matrix)
labels = af.labels_

# sum of similarities for each word
sum_of_similarities = numpy.sum(similarity_matrix,axis=1)


n = len(words)

V = numpy.zeros((max(labels)+1,400))
N = numpy.zeros(400)
W = [0]*n
for i in range(n):
    W[i] = model.wv[words[i]]
    V[labels[i]] = V[labels[i]] + W[i]

for i in range(max(labels)+1):
    V[i] = V[i] / numpy.linalg.norm(V[i])

max_similarity_word = [0]*(max(labels)+1)
max_dot_product_word = [0]*(max(labels)+1)
for i in range(max(labels)+1):
    cluster_indices = numpy.where(labels==i)[0];
    cluster_words = [words[l] for l in cluster_indices]
    cluster_words_total_similarity = [sum_of_similarities[l] for l in cluster_indices]
    cluster_words_dot_products = [numpy.dot(V[i],W[l]) for l in cluster_indices]
    max_similarity_word[i] = cluster_words[cluster_words_total_similarity.index(max(cluster_words_total_similarity))]
    max_dot_product_word[i] = cluster_words[cluster_words_dot_products.index(max(cluster_words_dot_products))]
    print(cluster_words)
    print(max_similarity_word[i],max_dot_product_word[i])

#create, sort, and save the clusters
L1 = [[words[i],labels[i],sum_of_similarities[i],max_similarity_word[labels[i]],max_dot_product_word[labels[i]]] for i in range(len(labels))]
L1.sort(key=lambda x: x[1])

results = pd.DataFrame(L1)
cols = ['Words', 'Clusters','Sum of Similarities','Word with maximum similarity(in cluster)','word with maximum dot product(in cluster)']
results.columns = cols
results.to_csv("clusters_h_1"+str(jj)+".csv")