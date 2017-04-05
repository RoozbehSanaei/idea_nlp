import pandas as pd
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy
import similarity_functions
import pre_processing
from cluster import cluster			




def w2v_word_clusters(selected_lemmas,c):

	"""

	import pickle
	output = open('data_pocket.pkl', 'wb')
	pickle.dump(selected_lemmas, output)

	import pickle
	pkl_file = open('data_pocket.pkl', 'rb')
	selected_lemmas = pickle.load(pkl_file)

	"""


	words = []
	for lemmas in selected_lemmas:
		words = words+lemmas



	# remove repetitive words or those that are not found in wikipedia vocabulary
	words = list(set(words))
	words = [w for w in words if w in similarity_functions.model.wv.vocab]

	tqdm(desc="Make Similarity Matrix")
	#make the similarity model
	similarity_matrix = numpy.zeros((len(words),len(words)))

	tqdm(desc="Remove Negative Correlations")
	#ignore negative corrolation
	from tqdm import trange
	for i in trange(len(similarity_matrix)):
		for j in range(len(similarity_matrix)):
			if (similarity_matrix[i][j]<0):
				similarity_matrix[i][j]=0


	labels = cluster(c,similarity_matrix)

	# sum of similarities for each word
	sum_of_similarities = numpy.sum(similarity_matrix,axis=1)


	n = len(words)

#    V = numpy.zeros((max(labels)+1,400))
#    N = numpy.zeros(400)

	W = [0]*n

	max_similarity_word = [0]*(max(labels)+1)
	max_dot_product_word = [0]*(max(labels)+1)
	cluster_words = [0]*(max(labels)+1)

	for i in range(max(labels)+1):
		cluster_indices = numpy.where(labels==i)[0];
		cluster_words[i] = [words[l] for l in cluster_indices]
		#cluster_words_total_similarity = [sum_of_similarities[l] for l in cluster_indices]
		#cluster_words_dot_products = [numpy.dot(V[i],W[l]) for l in cluster_indices]
		#print(cluster_words_total_similarity.index(max(cluster_words_total_similarity)),max(labels)+1, len(labels))
		#max_similarity_word[i] = cluster_words[cluster_words_total_similarity.index(max(cluster_words_total_similarity))]        
		#max_dot_product_word[i] = cluster_words[cluster_words_dot_products.index(max(cluster_words_dot_products))]

	#create, sort, and save the clusters
	return(cluster_words)

