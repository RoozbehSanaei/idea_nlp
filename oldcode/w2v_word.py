import pandas as pd
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy
import similarity_functions
import pre_processing
from cluster import cluster			



def calculate_matrix(selected_lemmas,model):
	words = []
	for lemmas in selected_lemmas:
		words = words+lemmas
	# remove repetitive words or those that are not found in wikipedia vocabulary
	words = list(set(words))
	words = [w for w in words if w in similarity_functions.model.wv.vocab]

	
	tqdm(desc="Make Similarity Matrix")
	#make the similarity model
	similarity_matrix = numpy.array([[model.similarity(w1, w2) for w1 in words] for w2 in tqdm(words)])


	tqdm(desc="Remove Negative Correlations")
	#ignore negative corrolation
	from tqdm import trange
	for i in trange(len(similarity_matrix)):
		for j in range(len(similarity_matrix)):
			if (similarity_matrix[i][j]<0):
				similarity_matrix[i][j]=0

	return words, similarity_matrix

def costs(cl1,M):
	IC = 0
	EC = 0
	cl = numpy.array(cl1)
	CS = [cl.tolist().count(i) for i in range(max(cl.tolist())+1)]
	for i in range(len(M)):
		for j in range(len(M)):
			if (cl[i]==cl[j]):
				IC = IC + M[i][j]/ (CS[cl[i]])
			else:
				EC = EC + M[i][j]/len(M)
	return IC,EC

def w2v_word( similarity_matrix,words,selected_lemmas,s,p):
	n = len(words)

	labels = cluster(similarity_matrix,p)

	# sum of similarities for each word
	sum_of_similarities = numpy.sum(similarity_matrix,axis=1)



#    V = numpy.zeros((max(labels)+1,400))
#    N = numpy.zeros(400)

	W = [0]*n

	max_similarity_word = [0]*(max(labels)+1)
	max_dot_product_word = [0]*(max(labels)+1)
	cluster_words = [0]*(max(labels)+1)

	for i in range(max(labels)+1):
		cluster_indices = numpy.where(labels==i)[0];
		cluster_words[i] = [words[l] for l in cluster_indices]

	#create, sort, and save the clusters

	if (s=='num_word_similarity'):
		M = [[similarity_functions.num_word_similarity(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
	elif (s=='total_set_similairy'):
		M = [[similarity_functions.total_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
	elif (s=='max_set_similairy'):
		M = [[similarity_functions.max_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
	elif (s=='vec_similairy'):
		M = [[similarity_functions.vec_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
	elif (s=='skipthoughts_similarity'):
		M = [[similarity_functions.vec_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]

	clusters =  [M[i].index(max(M[i])) for i in range(len(M))]

	return clusters

