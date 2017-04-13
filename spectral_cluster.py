from sklearn import cluster as clustering
import igraph
import numpy
import networkx as nx
import community

def spectral_cluster(M,p):
	clusters = [0]*len(M)

	c,a = p;


	try:	
			cl = clustering.SpectralClustering(n_clusters=c,affinity=a).fit(M)
			clusters = cl.labels_
	except:
			print ("==========Something went wrong here =============")
			pass


	return(clusters)



