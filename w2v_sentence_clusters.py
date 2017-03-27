import similarity_functions
import numpy
from sklearn import cluster
 
def w2v_sentence_clusters(sentences):
	cl_mat = numpy.zeros((len(sentences),len(sentences)))

	for sent1 in sentences:
	    for sent2  in sentences:
	        i1 = sentences.index(sent1)
	        i2 = sentences.index(sent2)
	        cl_mat[i1][i2] = similarity_functions.num_word_similarity(sent1,sent2)


	af = cluster.AffinityPropagation().fit(cl_mat)
	clusters = af.labels_
	return(clusters)
