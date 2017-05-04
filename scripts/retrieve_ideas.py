import pandas as pd
import numpy
panda_results = pd.read_csv("my_filtered.csv")
All_clusters = panda_results.ix[:,5]
All_clusters
clusters = [numpy.fromstring(All_clusters[i],dtype=int, sep=',') for i in range(len(All_clusters))]
file_names = panda_results.ix[:,4]
%

for i in range(10):
	f = file_names[i]
	df = pd.read_csv('/home/roozbeh/data/wiki/data/'+f+'.csv')
	df['clusters'] = clusters[i]
	df = df.sort_values(['clusters'], ascending=[True])
	df.to_csv('/home/roozbeh/data/wiki/data/'+str(i)+'.csv')

ids = [i for i in range(0,100)]

def uniform_random_alloc(n, ids):
	while True:
		inds = [random.randint(0,n-1) for _ in ids]
		bins = {i:set() for i in range(0,n)}
		for i, id_ in enumerate(ids):
			bins[inds[i]].add(id_)
		if all([len(bins[b])>0 for b in bins.keys()]):
			break
	return bins


d =  uniform_random_alloc(12,ids)
l =  numpy.zeros((1,len(ids)))