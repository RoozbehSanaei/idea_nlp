import numpy
import threading
import random
import math

def f(x):
	return [x[0]*x[0],x[1]*x[1],x[2]*x[2]]


inputs =  [[int(100*random.random()),int(100*random.random()),int(100*random.random())] for i in range(2000)]




def parallel_proc(f,inputs):

	results = [];
	
	number_of_threads=20
	l = len(inputs)
	m = math.ceil(l/number_of_threads)

	def ff(l):
		A = [f(x) for x in l]
		results.append(A)


	def ind(i):
		if (i<number_of_threads):
			return i*m
		elif (i==number_of_threads):
			return l


	threads = []
	for i in range(5):
		t = threading.Thread(target=ff, args=(inputs[ind(i):ind(i+1)],))
		threads.append(t)
		t.start()
 
 
	print('threads started')
	 
	for i in range(5):
	    threads[i].join()
	    print ('thread', i, 'finished')
	 
	 
	print('all finished')

	return results



print(parallel_proc(f,inputs))