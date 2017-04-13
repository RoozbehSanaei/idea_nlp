import w2v_word_spectral
import w2v_sentence_spectral
import pandas as pd
from tqdm import tqdm
import pre_processing
import similarity_functions
import numpy
import k_clique_cluster
import ipdb
import statistics
import spectral_cluster
import file_utils


def costs(cl1,M):
	IC = 0
	EC = 0
	cl = numpy.array(cl1)
	CS = [cl.tolist().count(i) for i in range(max(cl.tolist())+1)]
	for i in range(len(M)):
		for j in range(len(M)):
			if (cl[i]==cl[j]):
				IC = IC + M[i][j]/ (CS[cl[i]])
			else:
				EC = EC + M[i][j]/len(M)
	return IC,EC


def cross(v1,v_t):
	M = numpy.zeros((len(v_t),len(v_t)))
	for V1 in v1:
		for v_1 in V1:
			for v_2 in V1:
					M[v_t.index(v_1),v_t.index(v_2)]=1
	return(M)



def pre_process(df):
	tqdm.pandas(desc="Make Spacy Tokens")
	full_sentences = [f[2] for f in df.values]
	word_model = similarity_functions.model
	df['tokens'] = df.ix[:,2].progress_apply(lambda x: pre_processing.cleanPassage(x))    
	df['lemmas'] = df['tokens'].apply(lambda x: pre_processing.getLemmas(x))
	sentences = list(df['lemmas'])
	probs_cutoff_lower = pre_processing.findMeaningfulCutoffProbability([t for tok in df['tokens'] for t in tok])
	tqdm.pandas(desc="Remove redundant lemmas")
	selected_lemmas = df['tokens'].progress_apply(lambda x: pre_processing.makeNodelist(x,probs_cutoff_lower))
	words, word_similarity_matrix = w2v_word_spectral.calculate_matrix(selected_lemmas,word_model)
	return words,sentences,selected_lemmas, word_similarity_matrix

def evaluate(pre_processed_data,sentence_similarity_matrixs,e,parameters):
	words,sentences,selected_lemmas, word_similarity_matrix = pre_processed_data


	if (e=='word-based'):
		similarity_matrix = word_similarity_matrix
	else:
		similarity_matrix = sentence_similarity_matrix


	if (e=='word-based'):
		clusters,costs_ = w2v_word_spectral.w2v_word_spectral( similarity_matrix,words,selected_lemmas,s,parameters)
	else:
		clusters = spectral_cluster.spectral_cluster(similarity_matrix,parameters)
		costs_ = costs(clusters,similarity_matrix)


	print(clusters)
	print("number of clusters : ", len(set(clusters)))



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


	from collections import Counter
	cnt = Counter(clusters)
	m = statistics.median(cnt.values())

	return m,clusters,len(set(clusters)),float(numpy.sum(abs(M1-M2))),float(numpy.sum(numpy.multiply(M1,M2))),float(numpy.sum((M1>M2)*1)),float(numpy.sum((M2>M1)*1)),costs_


affinities =  ['nearest_neighbors', 'precomputed', 'rbf', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine']
similarities = ['num_word_similarity','total_set_similairy','max_set_similairy','vec_similairy','skipthoughts_similarity']
similarities = ['skipthoughts_similarity']
files = ['n50_de_rma']
n_clusters = [8,10,12,14,16,18,20,22,24]
element_type = ["word-based","sentence-based"]


import skipthoughts
sentence_model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(sentence_model)




total_number = len(similarities)*(len(files))*len(n_clusters)*len(element_type)*len(affinities)

results = []
restlts_with_matrix = []
import pandas as pd

j= 0

for f in files:
	file = '/home/roozbeh/data/wiki/data/'+f+'.csv'
	df = pd.read_csv(file)
	pre_processed_data = pre_process(df)
	words,sentences,selected_lemmas, word_similarity_matrix = pre_processed_data
	for s in similarities:
		sentence_similarity_matrix = w2v_sentence_spectral.calculate_matrix(sentences,s,encoder)
		file_utils.save_to_file(sentence_similarity_matrix,"similarity"+s+".pickl")
		for c in n_clusters:
			for a in affinities:
				for e in element_type:
					j = j + 1
					print(j,"/",total_number)
					print(a,c,s,e,f)
					
					
					m, cl,nn, dm,pm,fp,fn,costs_ = evaluate(pre_processed_data,s,e,[c,a])
					print(costs_)

					print("m: ",m)
					print(dm,pm)

					result = [j,s,c,nn,a,e,f,cl,dm,pm,m,fp,fn,costs_]
					#result_with_matrix = [j,s,c,nn,a,e,f,cl,dm,pm,m,fp,fn,similarity_matrix]
					results.append(result)
					#restlts_with_matrix.append(result_with_matrix)
					cols = ['index','Similarity_Type','Number of Clusters (input)','Number of clusters(output)','Affinity Type', 'Word/Sentence','File_Name','Clusters','Mismatch Metric','Match Metric','M','False Positive','False Negative','costs']
					panda_results = pd.DataFrame(results)
					panda_results.columns = cols
					panda_results.to_csv("/home/roozbeh/data/wiki/results/all_indexed.csv")

					df = pd.read_csv('/home/roozbeh/data/wiki/data/'+f+'.csv')
					df['clusters'] = cl
					df = df.sort_values(['clusters'], ascending=[True])
					df.to_csv('/home/roozbeh/data/wiki/results/spectral_'+str(j)+'.csv')

					
					print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
import pickle
f = open('store.pckl', 'wb')
pickle.dump(results, f)
f.close()
	




