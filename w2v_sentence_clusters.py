import similarity_functions
import numpy
from sklearn import cluster
import igraph
from cluster import cluster			

def w2v_sentence_clusters(sentences,s,c):
	cl_mat = numpy.zeros((len(sentences),len(sentences)))

	"""
	import pickle
	output = open('data_pocket.pkl', 'wb')
	pickle.dump([sentences,i], output)

	import pickle
	pkl_file = open('data_pocket.pkl', 'rb')
	[sentences,i] = pickle.load(pkl_file)

	"""

	for sent1 in sentences:
		for sent2  in sentences:
			i1 = sentences.index(sent1)
			i2 = sentences.index(sent2)

			if (s=='num_word_similarity'):
				cl_mat[i1][i2] = similarity_functions.num_word_similarity(sent1,sent2)
			elif (s=='total_set_similairy'):
				cl_mat[i1][i2] = similarity_functions.total_set_similairy(sent1,sent2)
			elif (s=='max_set_similairy'):
				cl_mat[i1][i2] = similarity_functions.max_set_similairy(sent1,sent2)
			elif (s=='vec_similairy'):
				cl_mat[i1][i2] = similarity_functions.vec_similairy(sent1,sent2)


	clusters = cluster(c,cl_mat)
	
	return(clusters)
