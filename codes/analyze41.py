
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


human_rated_files = [
'/home/roozbeh/data/wiki/Ideation Data/grouping similarity/n50_group_experienced_bc.csv',
'/home/roozbeh/data/wiki/Ideation Data/grouping similarity/n50_group_experienced_rma.csv',
'/home/roozbeh/data/wiki/Ideation Data/grouping similarity/n50sample_distributed_experienced_bc.csv',
'/home/roozbeh/data/wiki/Ideation Data/grouping similarity/n50sample_distributed_experienced_rma.csv',
'/home/roozbeh/data/wiki/Ideation Data/grouping similarity/n50sample_group_novice_bc.csv',
'/home/roozbeh/data/wiki/Ideation Data/grouping similarity/n50sample_group_novice_rma.csv'
]

files = [
'Distributed Experience and Novice (superset) clean',
'n1400 distributed novice','n1400 group experienced',
'n100 group novice',
'n100 distributed experienced'
]


outputbasepath = '/home/roozbeh/data/wiki/'

gn = pd.read_csv(human_rated_files[0])


	
tqdm.pandas(desc="Make Spacy Tokens")
gn['tokens'] = gn.ix[:,3].progress_apply(lambda x: cleanPassage(x))    
gn['lemmas'] = gn['tokens'].apply(lambda x: getLemmas(x))

probs_cutoff_lower = findMeaningfulCutoffProbability([t for tok in gn['tokens'] for t in tok])

tqdm.pandas(desc="Remove redundant tokens")
selected_tokens = gn['tokens'].progress_apply(lambda x: makeNodelist(x))
sentence_tokens = [list(set(tokens)) for tokens in selected_tokens]


tokens = []
for tokens_ in selected_tokens:
	tokens = tokens+tokens_
tokens = list(set(tokens))


limit_sets = [[]]
filter_sets = [[]]

n = 5;
s = 1+n+n**2
from tqdm import trange
for j in trange(s):
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


clusters=[]
m = int((len(limit_sets)-s)/n)

cluster_words = [[]]*(m)
for i in range(s,len(limit_sets)):
	c = int((i-s)/n)
	clusters.append([c]+[limit_sets[i][2]])
	cluster_words[c] = cluster_words[c] + [limit_sets[i][2]]

for i in range(m):
    cluster_indices = numpy.where(labels==i)[0];
    cluster_words[i] = [words[l] for l in cluster_indices]
    cluster_words_total_similarity = [sum_of_similarities[l] for l in cluster_indices]
    cluster_words_dot_products = [numpy.dot(V[i],W[l]) for l in cluster_indices]
    max_similarity_word[i] = cluster_words[cluster_words_total_similarity.index(max(cluster_words_total_similarity))]
    max_dot_product_word[i] = cluster_words[cluster_words_dot_products.index(max(cluster_words_dot_products))]
    print(cluster_words)
    print(max_similarity_word[i],max_dot_product_word[i])

def total_set_similairy(A,B):
    s = 0
    for w1 in A:
        for w2 in B:
            if (w2 in model.wv.vocab):
                s = s + model.similarity(w1,w2)
    s = s / (len(A)*len(B)+1)
    return s


def max_set_similairy(A,B):
    m = 0
    for w1 in A:
        for w2 in B:cc
            if (w2 in model.wv.vocab):
                if (model.similarity(w1,w2)>m):
                    m = model.similarity(w1,w2)
    return m



def vec_similairy(A,B):
    w_A = [model.wv[w] for w in A]
    w_B = [model.wv[w] for w in B]
    p = numpy.dot(w_A,w_B)/(numpy.linalg.norm(w_A)*numpy.linalg.norm(w_B))
    return p


import gensim
from gensim import corpora, models
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")    

M = [[len(set(sl) & set(cw)) for cw in cluster_words ] for sl in selected_tokens]

M = [[total_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_tokens]

M = [[max_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_tokens]


clusters =  [M[i].index(max(M[i])) for i in range(len(M))]


L1 = [[clusters[i],gn['ID'][i]] for i in range(len(selected_tokens))]
L1.sort(key=lambda x: x[0])

results = pd.DataFrame(L1)
results.to_csv("clusters_w2v_nc"+str(jj)+".csv")

