import numpy
from multiprocessing.pool import ThreadPool
import random


def f(x):
	return [x[0]*x[0],x[1]*x[1],x[2]*x[2]]

inputs =  [[int(100*random.random()),int(100*random.random()),int(100*random.random())] for i in range(number_of_threads)]


def parallel_proc(inputs,f):
	number_of_threads = 20;
	pool = ThreadPool(processes=number_of_threads+2)

	async_result = [0]*number_of_threads;
	return_val = [0]*number_of_threads;

	for i in range(number_of_threads):
		print("thread started : ", i)
		async_result[i] = pool.apply_async(f, (inputs[i],)) # tuple of args for foo


	for i in range(number_of_threads):
		print("thread ended : ", i)
		return_val[i] = async_result[i].get() 


	return_vals = [return_val[i] for i in range(number_of_threads)];


	results=numpy.concatenate(return_vals, axis=0);
	return results

parallel_proc(inputs)