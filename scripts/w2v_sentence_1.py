import numpy
from multiprocessing.pool import ThreadPool
import random
import math
import pre_processing
import pandas as pd
import similarity_functions
import file_utils

import skipthoughts
sentence_model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(sentence_model)


def parallel_proc(f,inputs):
	
	number_of_threads=20
	l = len(inputs)
	m = math.ceil(l/number_of_threads)

	def ff(l):
		return [f(x) for x in l]

	def ind(i):
		if (i<number_of_threads):
			return i*m
		elif (i==number_of_threads):
			return l

	async_result = [pool.apply_async(ff, (inputs[ind(i):ind(i+1)],)) for i in range(number_of_threads)]

	return_vals = [async_result[i].get() for i in range(number_of_threads)]

	results=numpy.concatenate(return_vals, axis=0);
	return results

def calculate_matrix(sentences,s):
	sentences_strs = [" ".join(sent) for sent in sentences]
	sentence_vectors = encoder.encode(sentences_strs)


	inputs = [0 for i in range(l*l)]

	k = 0;
	for i in range(l):
		for j in range(l):
			inputs[k] = [i,j]
			k = k + 1;



	mat = [[0 for i in range(l)] for j in range(l)]


	global counter; counter = 0 

	def f(x):
		global counter
		i = x[0]
		j = x[1]
		sent_i = sentences[i]
		sent_j = sentences[j]
		
		if (s=='skipthoughts_similarity'):
			mat[i][j] = numpy.dot(sentence_vectors[i],sentence_vectors[j])
		elif (s=='skipthoughts_similarity_N'):
			mat[i][j] = numpy.dot(sentence_vectors[i],sentence_vectors[j])/(numpy.linalg.norm(sentence_vectors[i])*numpy.linalg.norm(sentence_vectors[j])+0.000001)

		counter = counter + 1
		print(counter)

	parallel_proc(f,inputs)

file_utils.save_to_file(mat,"skipthoughts_similarity.pickl")
