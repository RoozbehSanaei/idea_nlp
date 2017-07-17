#all imports
import w2v_word
import pandas as pd
from tqdm import tqdm
import pre_processing
import similarity_functions
import numpy
import k_clique_cluster
import ipdb
import statistics
import cluster
import file_utils
from collections import Counter
import pandas as pd
import graph_utils

def evaluate(pre_processed_data,sentence_similarity_matrixs,e,parameters,df):
	words,sentences,selected_lemmas, word_similarity_matrix = pre_processed_data


	if (e=='word-based'):
		clusters = w2v_word.w2v_word( word_similarity_matrix,words,selected_lemmas,s,parameters)
	else:
		clusters = cluster.cluster(sentence_similarity_matrix,parameters)


	print("number of clusters : ", len(set(clusters)))



	L = df.ix[:,1].tolist()
	L0 = df.ix[:,1].tolist()
	clusters_adapted = [clusters[L.index(x)] for x in L0 if x in L]
	
	M1 = graph_utils.cross1(clusters_adapted)
	M2 = graph_utils.cross1(df.ix[:,0].tolist())


	cnt = Counter(clusters)
	m = statistics.median(cnt.values())

	return m,clusters,len(set(clusters)),float(numpy.sum(abs(M1-M2))),float(numpy.sum(numpy.multiply(M1,M2))),float(numpy.sum((M1>M2)*1)),float(numpy.sum((M2>M1)*1))


similarities = ['vec_similairy','skipthoughts_similarity','num_word_similarity','total_set_similairy','max_set_similairy']
files = ['n1400_dn']
clusterings_algs = ["louvain","spectralـnearest_neighbors","spectralـprecomputed","spectralـrbf","spectralـsigmoid","spectralـpolynomial","spectralـpoly","spectralـlinear","agglomerative_clustering","affinity_propagation","kmeans","birch","ward_clustering"]
element_type = ["sentence-based"]

total_number = len(similarities)*(len(files))*len(element_type)

results = []
restlts_with_matrix = []

j= 0

for f in files:
	file = '/home/roozbeh/data/wiki/data/'+f+'.csv'
	df = pd.read_csv(file)
	pre_processed_data = pre_processing.pre_process(df)
	words,sentences,selected_lemmas, word_similarity_matrix = pre_processed_data
	#pre_processed_data = file_utils.load_from_file('pre_process.pickl')
	file_utils.save_to_file(pre_processed_data,'pre_process.pickl')
	for s in similarities:
		sentence_similarity_matrix = file_utils.load_from_file(s+".pickl")
		for e in element_type:
			for alg in clusterings_algs:
				if clusterings_algs.index(alg)>14:
					n_clusters = [100]
				else:
					n_clusters = [20,30,40,50,60,70,80,90,100]
					n_clusters = [100]
					
				for c in n_clusters:
					j = j + 1
					print(j,alg,s,c,e,f)
					
					
					m, cl,nn, dm,pm,fp,fn = evaluate(pre_processed_data,s,e,[alg,c],df)

					print("m: ",m)

					result = [alg,j,s,c,nn,e,f,cl,dm,pm,m,fp,fn]
					print(result)
					#result_with_matrix = [j,s,c,nn,a,e,f,cl,dm,pm,m,fp,fn,similarity_matrix]
					results.append(result)
					#restlts_with_matrix.append(result_with_matrix)
					cols = ['algorithm','index','Similarity_Type','Number of Clusters (input)','Number of clusters(output)', 'Word/Sentence','File_Name','Clusters','Mismatch Metric','Match Metric','M','False Positive','False Negative']
					panda_results = pd.DataFrame(results)
					panda_results.columns = cols
					panda_results.to_csv("/home/roozbeh/data/wiki/results/all_indexed.csv")

					df = pd.read_csv('/home/roozbeh/data/wiki/data/'+f+'.csv')
					df['clusters'] = cl
					df = df.sort_values(['clusters'], ascending=[True])
					df.to_csv('/home/roozbeh/data/wiki/results/spectral_'+str(j)+'.csv')

					
					print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
import pickle
f = open('results.pckl', 'wb')
pickle.dump(results, f)
f.close()
	




