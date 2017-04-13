import similarity_functions
import numpy
from sklearn import cluster
import igraph
from spectral_cluster import spectral_cluster			
from tqdm import tqdm


def calculate_matrix(sentences,s,encoder):
	cl_mat = numpy.zeros((len(sentences),len(sentences)))
	for sent1 in tqdm(sentences):
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
			elif (s=='skipthoughts_similarity'):			
				cl_mat[i1][i2] = similarity_functions.skipthoughts_similarity(sent1,sent2,encoder)	
	return cl_mat
