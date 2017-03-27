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

import glob
print(sorted(glob.glob("final_analysis_sets/*.csv")))


human_rated_file = [
'final_analysis_sets/n100 distributed experienced - human v3.csv', 
'final_analysis_sets/n100 distributed experienced - reconciled.csv', 
'final_analysis_sets/n100 distributed experienced.csv', 
'final_analysis_sets/n100 distributed novice - human v3.csv', 
'final_analysis_sets/n100 distributed novice - reconciled.csv', 
'final_analysis_sets/n100 distributed novice.csv', 
'final_analysis_sets/n100 group experienced - human v3.csv', 
'final_analysis_sets/n100 group experienced - reconciled.csv', 
'final_analysis_sets/n100 group experienced.csv', 
'final_analysis_sets/n100 group novice - human v3.csv',
 'final_analysis_sets/n100 group novice - reconciled.csv', 
 'final_analysis_sets/n100 group novice.csv',
  'final_analysis_sets/n50 distributed experienced - brad.csv', 
  'final_analysis_sets/n50 distributed experienced - ryan.csv', 
  'final_analysis_sets/n50 distributed novice - brad.csv', 
  'final_analysis_sets/n50 distributed novice - ryan.csv', 
  'final_analysis_sets/n50 group experienced - brad.csv', 
  'final_analysis_sets/n50 group experienced - ryan.csv', 
  'final_analysis_sets/n50 group novice - brad.csv', 
  'final_analysis_sets/n50 group novice - ryan.csv'
  ]




jj = 0

outputbasepath = '/home/roozbeh/data/wiki/'

gn = pd.read_csv(human_rated_files[jj])

    
tqdm.pandas(desc="Make Spacy Tokens")
gn['tokens'] = gn.ix[:,3].progress_apply(lambda x: cleanPassage(x))    
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
af = AffinityPropagation(preference=-4.5).fit(similarity_matrix)
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
cluster_words = [0]*(max(labels)+1)

for i in range(max(labels)+1):
    cluster_indices = numpy.where(labels==i)[0];
    cluster_words[i] = [words[l] for l in cluster_indices]
    cluster_words_total_similarity = [sum_of_similarities[l] for l in cluster_indices]
    cluster_words_dot_products = [numpy.dot(V[i],W[l]) for l in cluster_indices]
    max_similarity_word[i] = cluster_words[cluster_words_total_similarity.index(max(cluster_words_total_similarity))]
    max_dot_product_word[i] = cluster_words[cluster_words_dot_products.index(max(cluster_words_dot_products))]

#create, sort, and save the clusters

def total_set_similairy(A,B):
    s = 0
    for w1 in A:
        for w2 in B:
            if ((w1 in model.wv.vocab) & (w2 in model.wv.vocab)):
                s = s + model.similarity(w1,w2)
    s = s / (len(A)*len(B)+1)
    return s

def max_set_similairy(A,B):
    m = 0
    for w1 in A:
        for w2 in B:
            if ((w1 in model.wv.vocab) & (w2 in model.wv.vocab)):
                if (model.similarity(w1,w2)>m):
                    m = model.similarity(w1,w2)
    return m



def vec_similairy(A,B):
    p = 0
    w_A = sum([model.wv[w] for w in A])
    w_B = sum([model.wv[w] for w in B if w in model.wv.vocab])
    if ((numpy.linalg.norm(w_A)!=0) & (numpy.linalg.norm(w_B)!=0)):
        p = numpy.dot(w_A,w_B)/(numpy.linalg.norm(w_A)*numpy.linalg.norm(w_B)+0.000001)
    return p

def num_word_similarity(sent1,sent2):
    l = len(set(sent1) & set(sent2))
    return l


M = [[num_word_similarity(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]

M = [[total_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]

#M = [[max_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]

#M = [[vec_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]


clusters =  [M[i].index(max(M[i])) for i in range(len(M))]


L1 = [[clusters[i],gn['ID'][i]] for i in range(len(selected_lemmas))]
L1.sort(key=lambda x: x[0])

results = pd.DataFrame(L1)
results.to_csv("clusters_w2v_h"+str(jj)+".csv")



#analyzing brad data


print ("analyzing brad data")
ii = 0
bc = pd.read_csv(human_rated_files[jj])
rma = pd.read_csv(human_rated_files[jj+1])

    
bc['tokens'] = bc.ix[:,3].progress_apply(lambda x: cleanPassage(x))    
bc['lemmas'] = bc['tokens'].apply(lambda x: getLemmas(x))
IDs = bc.ix[:,2]

sentences = list(bc['lemmas'])




cl_mat = numpy.zeros((len(sentences),len(sentences)))

for sent1 in sentences:
    for sent2  in sentences:
        i1 = sentences.index(sent1)
        i2 = sentences.index(sent2)
        cl_mat[i1][i2] = num_word_similarity(sent1,sent2)
cl_mat = numpy.sign(cl_mat-1)


data_ryan = [[rma.ix[:,0][i],str(rma.ix[:,1][i])] for i in range(len(clusters))]
data_bc = [[bc.ix[:,1][i],str(bc.ix[:,2][i])] for i in range(len(clusters))]
data_cl = [[str(clusters[i]),str(IDs[i])] for i in range(len(clusters))]


def david_dict(data):
    cats = [d[0] for d in data if "noise" not in d[0]]
    dout = {c:set([int(d[1]) for d in data if d[0]==c]) for c in cats}
    #dout['noise'] = set([int(d[1]) for d in data if 'noise' in d[0]])
    return dout


import jaccard                

d1 = david_dict(data_ryan)
d2 = david_dict(data_bc)
d3 = david_dict(data_cl)
jaccard.greedy_match_sets(d1, d3)
jaccard.greedy_match_sets(d1, d2)
