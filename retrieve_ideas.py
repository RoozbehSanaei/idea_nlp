import pandas as pd
import numpy
panda_results = pd.read_csv("ryan_selected_clusters.csv")
All_clusters = panda_results.ix[:,6]
All_clusters
clusters = [numpy.fromstring(All_clusters[i],dtype=int, sep=',') for i in range(3)]
file_names = panda_results.ix[:,5]

for i in range(3):
	f = file_names[i]
	df = pd.read_csv('/home/roozbeh/data/wiki/data/'+f+'.csv')
	df['clusters'] = clusters[i]
	df = df.sort_values(['clusters'], ascending=[True])
	df.to_csv('/home/roozbeh/data/wiki/data/'+str(i)+'.csv')
