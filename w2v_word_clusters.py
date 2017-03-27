import pandas as pd
import string
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import similarity_functions
import pre_processing



def w2v_word_clusters(selected_lemmas):

    words = []
    for lemmas in selected_lemmas:
    	words = words+lemmas



    # remove repetitive words or those that are not found in wikipedia vocabulary
    words = list(set(words))
    words = [w for w in words if w in similarity_functions.model.wv.vocab]

    tqdm(desc="Make Similarity Matrix")
    #make the similarity model
    similarity_matrix = [[similarity_functions.model.similarity(w1, w2) for w1 in words] for w2 in tqdm(words)]


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
        W[i] = similarity_functions.model.wv[words[i]]
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
    return(cluster_words)

