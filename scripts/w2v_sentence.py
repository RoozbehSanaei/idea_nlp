import similarity_functions
import numpy
from sklearn import cluster
import igraph
from cluster import cluster			
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool 
from functools import partial
from parallel_utils1 import parallel_proc




def calculate_matrix(sentences,s):
	
	pool = ThreadPool(20)
	#cl_mat = pool.map(partial(list_similarity,sentences = sentences,sim_metric = s), sentences)

	M = [0]*len(sentences)


	if (s=='num_word_similarity'):
		for i in range(len(sentences)):
			M[i] = parallel_proc(partial(similarity_functions.num_word_similarity,sentences[i]),sentences) 
	elif (s=='total_set_similairy'):
		for i in range(len(sentences)):
			M[i] = parallel_proc(partial(similarity_functions.total_set_similairy,sentences[i]),sentences) 
	elif (s=='max_set_similairy'):
		for i in range(len(sentences)):
			M[i] = parallel_proc(partial(similarity_functions.max_set_similairy,sentences[i]),sentences) 
	elif (s=='vec_similairy'):
		for i in range(len(sentences)):
			M[i] = parallel_proc(partial(similarity_functions.vec_similairy,sentences[i]),sentences) 			
	elif (s=='skipthoughts_similarity'):
		for i in range(len(sentences)):
			M[i] = parallel_proc(partial(similarity_functions.skipthoughts_similarity,sentences[i]),sentences) 				
	


	M = numpy.array(M)
	return M
