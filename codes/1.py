import csv
import gensim
import numpy
import networkx as nx
import hdbscan
import community



#read ryan csv
with open('superset.csv') as input_file:
	rows = csv.reader(input_file, delimiter=';')
	res = list(zip(*rows))


"""



#Matrix form Ryan Graph
M = [list(map(float, res[i][1:len(res)])) for i in range(1,len(res))];


#list of WORD-POSTAGs
words = res[0][1:len(res)];
words = [word.replace(' ', '-') for word in words];

"""





#Similarity Model from wikipedia
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")    
#model = gensim.models.KeyedVectors.load_word2vec_format('/home/roozbeh/data/wiki/GoogleNews-vectors-negative300.bin', binary=True);
#extract words from ryan csv

words = [(w.split(' '))[0] for w in res[0] if (w.split(' '))[0] in model.wv.vocab]
words = list(set(words))


#construct similarity matrix
M = [[model.similarity(w1, w2) for w1 in words] for w2 in words]

for i in range(len(words)):
	for j in range(len(words)):
		if (M[i][j]<0):
			M[i][j]=0

#find WORDS that are in wikipedia vocabulary







#find similar words
"""S = numpy.argsort(-MAT[words.index("toll")])  
T = -numpy.sort(-MAT[words.index("toll")])  """



#find words which are not similar to the rest at all
"""MAT_second_best = numpy.array([numpy.sort(M)[-2] for M in MAT])
numpy.where((MAT_second_best<0.45)&(0.4<MAT_second_best))[0][0]"""



# measure centralitry
"""MAT= numpy.array(M)
G=nx.from_numpy_matrix(MAT)
a=nx.betweenness_centrality(G)
c = [x[1] for x in a.items()]
s = numpy.argsort(c)"""


#Clusering through Affinity Propagation
from sklearn.cluster import AffinityPropagation
af = AffinityPropagation(preference=-50).fit(M)
labels = af.labels_
s = numpy.where(labels==0)[0]

"""
from sklearn.cluster import MeanShift, estimate_bandwidth
bandwidth = estimate_bandwidth(numpy.array(M), quantile=0.2, n_samples=len(M))
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(M)
labels = ms.labels_

#Clusering through HDBScan
clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=False, leaf_size=40,
    metric='euclidean', min_cluster_size=5, min_samples=None, p=None)

cluster_labels = clusterer.fit_predict(M)

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(M)
labels = db.labels_

from mcl_clustering import mcl
M, clusters = mcl(numpy.array(M))


from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering()
model.fit(M)
labels=model.labels_


G=nx.from_numpy_matrix(M)
partition = community.best_partition(M,resolution=0.1)
partition_i = partition.items();
"""

L = [[words[i],labels[i]] for i in range(len(labels))]

with open("clusters0.csv", "wb") as output_file:
	writer = csv.writer(output_file, delimiter=';')
	writer.writerows(L)