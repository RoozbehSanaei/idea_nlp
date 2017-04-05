from sklearn import cluster as clustering
import igraph
import numpy
import networkx as nx
import community

def cluster(c,M):
	clusters = [0]*len(M)

	try:
		if c == "community_walktrap_2":
			g = igraph.Graph.Adjacency(M.tolist(),mode = "undirected")
			g = g.simplify();
			d = g.community_walktrap(steps=2)
			clusters = d.as_clustering().membership			

		if c == "community_walktrap":
			g = igraph.Graph.Adjacency(M.tolist(),mode = "undirected")
			g = g.simplify();
			d = g.community_walktrap()
			clusters = d.as_clustering().membership			

		if c == "community_edge_betweenness":
			g = igraph.Graph.Adjacency(M.tolist(),mode = "undirected")
			g = g.simplify();
			d = g.community_edge_betweenness()
			clusters = d.as_clustering().membership			


		if c == "community_fastgreedy":
			g = igraph.Graph.Adjacency(M.tolist(),mode = "undirected")
			g = g.simplify();
			d = g.community_fastgreedy()
			clusters = d.as_clustering().membership			

		if c == "community_multilevel":
			g = igraph.Graph.Adjacency(M.tolist(),mode = "undirected")
			g = g.simplify();
			d = g.community_multilevel()
			clusters = d.membership			

		if c == "community_spinglass":
			g = igraph.Graph.Adjacency(M.tolist(),mode = "undirected")
			g = g.simplify();
			d = g.community_spinglass()
			clusters = d.membership			

		if c == "community_leading_eigenvector":
			g = igraph.Graph.Adjacency(M.tolist(),mode = "undirected")
			g = g.simplify();
			d = g.community_leading_eigenvector()
			clusters = d.membership			

		if c == "louvain":
			g = nx.from_numpy_matrix(M)
			p = community.best_partition(g)
			clusters = [p[i] for i in p.keys()]	
				

		if c == "KMeans_8":
			cl = clustering.KMeans(n_clusters=8).fit(M)
			clusters = cl.labels_

		if c == "KMeans_10":
			cl = clustering.KMeans(n_clusters=10).fit(M)
			clusters = cl.labels_

		if c == "KMeans_12":
			cl = clustering.KMeans(n_clusters=12).fit(M)
			clusters = cl.labels_

		if c == "Birch_8":
			cl = clustering.Birch(n_clusters=8).fit(M)
			clusters = cl.labels_

		if c == "Birch_10":
			cl = clustering.Birch(n_clusters=10).fit(M)
			clusters = cl.labels_

		if c == "Birch_12":
			cl = clustering.Birch(n_clusters=12).fit(M)
			clusters = cl.labels_

		if c == "Spectral_Clustering_8":
			cl = clustering.SpectralClustering(n_clusters=8).fit(M)
			clusters = cl.labels_

		if c == "Spectral_Clustering_10":
			cl = clustering.SpectralClustering(n_clusters=10).fit(M)
			clusters = cl.labels_

		if c == "Spectral_Clustering_12":
			cl = clustering.SpectralClustering(n_clusters=12).fit(M)
			clusters = cl.labels_

		if c == "Ward_Clustering_8":
			cl = clustering.AgglomerativeClustering(n_clusters=8).fit(M)
			clusters = cl.labels_

		if c == "Ward_Clustering_10":
			cl = clustering.AgglomerativeClustering(n_clusters=10).fit(M)
			clusters = cl.labels_

		if c == "Ward_Clustering_12":
			cl = clustering.AgglomerativeClustering(n_clusters=12).fit(M)
			clusters = cl.labels_

		if c == "Agglomerative_Clustering_8":
			cl = clustering.AgglomerativeClustering(n_clusters=8, linkage='complete').fit(M)
			clusters = cl.labels_

		if c == "Agglomerative_Clustering_10":
			cl = clustering.AgglomerativeClustering(n_clusters=10, linkage='complete').fit(M)
			clusters = cl.labels_

		if c == "Agglomerative_Clustering_12":
			cl = clustering.AgglomerativeClustering(n_clusters=12, linkage='complete').fit(M)
			clusters = cl.labels_

		if c == "AffinityPropagation":
			cl = clustering.AffinityPropagation().fit(M)
			clusters = cl.labels_

	except:
		print ("==========Something went wrong here =============")
		pass

	return(clusters)



