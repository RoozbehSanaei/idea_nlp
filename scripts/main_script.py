import w2v_word_clusters
import pandas as pd
from tqdm import tqdm
import pre_processing
import similarity_functions
import numpy
import k_clique_cluster
import ipdb

def cross(v1,v_t):
	M = numpy.zeros((len(v_t),len(v_t)))
	for V1 in v1:
		for v_1 in V1:
			for v_2 in V1:
					M[v_t.index(v_1),v_t.index(v_2)]=1
	return(M)


def evaluate(s,e,f,c):


	file = '/home/roozbeh/data/wiki/data/'+f+'.csv'


	df = pd.read_csv(file)


	tqdm.pandas(desc="Make Spacy Tokens")
	full_sentences = [f[2] for f in df.values]

	df['tokens'] = df.ix[:,2].progress_apply(lambda x: pre_processing.cleanPassage(x))    
	df['lemmas'] = df['tokens'].apply(lambda x: pre_processing.getLemmas(x))
	sentences = list(df['lemmas'])
	probs_cutoff_lower = pre_processing.findMeaningfulCutoffProbability([t for tok in df['tokens'] for t in tok])
	tqdm.pandas(desc="Remove redundant lemmas")
	selected_lemmas = df['tokens'].progress_apply(lambda x: pre_processing.makeNodelist(x,probs_cutoff_lower))


	if (e=='word-based'):

		if (c=="k-clique-4"):
			cluster_words = k_clique_cluster.get_cluster(f,4)
		elif (c=="k-clique-6"):
			cluster_words = k_clique_cluster.get_cluster(f,6)		
		elif (c=="k-clique-8"):
			cluster_words = k_clique_cluster.get_cluster(f,8)
		else:
			cluster_words = w2v_word_clusters.w2v_word_clusters(selected_lemmas,c)


		if (s=='num_word_similarity'):
			M = [[similarity_functions.num_word_similarity(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
		elif (s=='total_set_similairy'):
			M = [[similarity_functions.total_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
		elif (s=='max_set_similairy'):
			M = [[similarity_functions.max_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
		elif (s=='vec_similairy'):
			M = [[similarity_functions.vec_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]

		clusters =  [M[i].index(max(M[i])) for i in range(len(M))]

	else:
		import w2v_sentence_clusters
		clusters = w2v_sentence_clusters.w2v_sentence_clusters(sentences,s,c)




	#result_file = open('results/'+f+'_'+str(i)+'_'+str(j)+'.txt', "w")
	print(clusters)
	print("number of clusters : ", len(set(clusters)))


	"""
	import pickle
	output = open('data_pocket.pkl', 'wb')
	pickle.dump(selected_lemmas, output)
	"""

	"""
	import pickle
	output = open('data_pocket.pkl', 'wb')
	pickle.dump([v1,v2,v_t], output)
	"""

	#jaccard evaulaton
	import jaccard                

	data_df = [[str(df.ix[:,0][i]),str(df.ix[:,1][i])] for i in range(len(clusters))]
	data_cl = [[str(clusters[i]),str(df.ix[:,1][i])] for i in range(len(clusters))]
	d1 = pre_processing.david_dict(data_df)
	d2 = pre_processing.david_dict(data_cl)
	
	v1 = list(d1.values())
	v2 = list(d2.values())
	v_t = list(df.ix[:,1])
	
	M1 = cross(v1,v_t)
	M2 = cross(v2,v_t)



	#qipdb.set_trace()
	return clusters,len(set(clusters)),float(numpy.linalg.norm(M1-M2)),float(jaccard.greedy_match_sets(d1, d2)/len(selected_lemmas))
	#result_file.close()


	"""
	#save to file
	L1 = [[clusters[i],df.ix[:,1][i]] for i in range(len(selected_lemmas))]
	L1.sort(key=lambda x: x[0])
	results = pd.DataFrame(L1)
	results.to_csv("clusters_w2v_h")
	"""

similarities = ['num_word_similarity','total_set_similairy','max_set_similairy','vec_similairy']
files3 = ['n50_de_bc','n50_de_rma','n50_ge_bc','n50_ge_rma','n50_gn_bc','n50_gn_rma','n50_dn_bc','n50_dn_rma']
files1 = ['n100_de','n100_dn','n100_ge','n100_gn']
files2 = ['n100_de_v3','n100_dn_v3','n100_ge_v3','n100_gn_v3']
clusterings_algs = ["AffinityPropagation","louvain","k-clique-4","k-clique-6","k-clique-8","KMeans_8","KMeans_10","KMeans_12","Birch_8","Birch_10","Birch_12",
"Spectral_Clustering_8","Spectral_Clustering_10","Spectral_Clustering_12","Ward Clustering_8","Ward Clustering_10","Ward Clustering_12"
"Agglomerative_Clustering_8","Agglomerative_Clustering_10","Agglomerative_Clustering_12","community_fastgreedy",
"community_multilevel","community_spinglass","community_leading_eigenvector","community_edge_betweenness","community_walktrap_2","community_walktrap"]
element_type = ["word-based","sentence-based"]


total_number = len(similarities)*(len(files1)+len(files2)+len(files3))*len(clusterings_algs)*len(element_type)
all_files = [files1,files2,files3]

results = []
import pandas as pd

j= 0

for i in range(3):
	for f in all_files[i]:
		for s in similarities:
				for e in element_type:
					for c in clusterings_algs:
						j = j + 1
						print(j,"/",total_number)
						print(c)
						print(s)
						print(e)
						print(f)
						cl,nn, rm, dm = evaluate(s,e,f,c)
						
						"""
						if ev>=0.5:
							n5 = n5 + 1
						if ev>=0.6:
							n6 = n6 + 1
						if ev>=0.7:
							n7 = n7 + 1
						"""

						print(dm)
						print(rm)

						result = [c,s,e,f,cl,nn,rm,dm]
						results.append(result)
						cols = ['Clustering algorithm','Similarity_Type', 'Node_Type','File_Name','Clusters','Number of clusters','Roozbeh Metric','David Metric']
						panda_results = pd.DataFrame(results)
						panda_results.columns = cols
						panda_results.to_csv("all_clusters.csv")
						print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')




