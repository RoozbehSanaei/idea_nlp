import numpy
from multiprocessing.pool import ThreadPool
import random
import math

def f(x):
	return x[0]*x[0]+x[1]*x[1]+x[2]*x[2]

pool = ThreadPool(processes=22)


inputs =  [[int(100*random.random()),int(100*random.random()),int(100*random.random())] for i in range(2000)]


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


	for i in range(number_of_threads):
		print("thread started : ", i)
		async_result = [pool.apply_async(ff, (inputs[ind(i):ind(i+1)],)) for i in range(number_of_threads)]


	for i in range(number_of_threads):
		print("thread ended : ", i)
		return_vals = [async_result[i].get() for i in range(number_of_threads)]

	results=numpy.concatenate(return_vals, axis=0);
	return results

#print(parallel_proc(f,inputs))