import similarity_functions
import numpy
from sklearn import cluster
 
def w2v_sentence_clusters(sentences,i):
	cl_mat = numpy.zeros((len(sentences),len(sentences)))

	for sent1 in sentences:
		for sent2  in sentences:
			i1 = sentences.index(sent1)
			i2 = sentences.index(sent2)

			if (i==1):
				cl_mat[i1][i2] = similarity_functions.num_word_similarity(sent1,sent2)
			elif (i==2):
				cl_mat[i1][i2] = similarity_functions.total_set_similairy(sent1,sent2)
			elif (i==3):
				cl_mat[i1][i2] = similarity_functions.max_set_similairy(sent1,sent2)
			elif (i==4):
				cl_mat[i1][i2] = similarity_functions.vec_similairy(sent1,sent2)



	af = cluster.AffinityPropagation().fit(cl_mat)
	clusters = af.labels_
	return(clusters)
