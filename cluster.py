from sklearn import cluster as clustering
import igraph
import numpy
import networkx as nx
import community

def cluster(M,p):
	clusters = [0]*len(M)
	threshold_for_bug = 0.00000001
	M = numpy.array(M)
	M[M<threshold_for_bug]=threshold_for_bug
	M = M.tolist()

	alg,c = p;

	try:
		if alg == "louvain":
			g = nx.from_numpy_matrix(numpy.array(M))
			p = community.best_partition(g)
			clusters = [p[i] for i in p.keys()]	
				
		if alg == "kmeans":
			cl = clustering.KMeans(n_clusters=c).fit(M)
			clusters = cl.labels_

		if alg == "birch":
			cl = clustering.Birch(n_clusters=c).fit(M)
			clusters = cl.labels_


		if alg == "ward_clustering":
			cl = clustering.AgglomerativeClustering(n_clusters=c).fit(M)
			clusters = cl.labels_

		if alg == "agglomerative_clustering":
			cl = clustering.AgglomerativeClustering(n_clusters=c, linkage='complete').fit(M)
			clusters = cl.labels_

		if alg == "affinity_propagation":
			cl = clustering.AffinityPropagation().fit(M)
			clusters = cl.labels_

		if alg == "spectralـnearest_neighbors":
			cl = clustering.SpectralClustering(n_clusters=c,affinity='nearest_neighbors').fit(M)
			clusters = cl.labels_

		if alg == "spectralـprecomputed":
			cl = clustering.SpectralClustering(n_clusters=c,affinity='precomputed').fit(M)
			clusters = cl.labels_

		if alg == "spectralـrbf":
			cl = clustering.SpectralClustering(n_clusters=c,affinity='rbf').fit(M)
			clusters = cl.labels_

		if alg == "spectralـsigmoid":
			cl = clustering.SpectralClustering(n_clusters=c,affinity='sigmoid').fit(M)
			clusters = cl.labels_

		if alg == "spectralـpolynomial":
			cl = clustering.SpectralClustering(n_clusters=c,affinity='polynomial').fit(M)
			clusters = cl.labels_

		if alg == "spectralـpoly":
			cl = clustering.SpectralClustering(n_clusters=c,affinity='poly').fit(M)
			clusters = cl.labels_

		if alg == "spectralـlinear":
			cl = clustering.SpectralClustering(n_clusters=c,affinity='linear').fit(M)
			clusters = cl.labels_

		if alg == "spectralـcosine":
			cl = clustering.SpectralClustering(n_clusters=c,affinity='cosine').fit(M)
			clusters = cl.labels_
	except :
		print("error occurred")

		pass	
	return(clusters)



