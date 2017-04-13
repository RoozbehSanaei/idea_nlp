import w2v_word
import w2v_sentence
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


def cross1(cl1):
	M = numpy.zeros((len(cl1),len(cl1)))
	for i in range(len(cl1)):
		for j in range(len(cl1)):
			if (cl1[i]==cl1[j]):
				M[i,j] = 1
	return(M)

def pre_process(df):
	tqdm.pandas(desc="Make Spacy Tokens")
	#full_sentences = [f[1] for f in df.values]

	df['tokens'] = df.ix[:,2].progress_apply(lambda x: pre_processing.cleanPassage(x))    
	df['lemmas'] = df['tokens'].apply(lambda x: pre_processing.getLemmas(x))
	sentences = list(df['lemmas'])
	probs_cutoff_lower = pre_processing.findMeaningfulCutoffProbability([t for tok in df['tokens'] for t in tok])
	tqdm.pandas(desc="Remove redundant lemmas")
	selected_lemmas = df['tokens'].progress_apply(lambda x: pre_processing.makeNodelist(x,probs_cutoff_lower))
	words, word_similarity_matrix = w2v_word.calculate_matrix(selected_lemmas)
	return words,sentences,selected_lemmas, word_similarity_matrix

def evaluate(df0,df,pre_processed_data,sentence_similarity_matrixs,e,alg):
	words,sentences,selected_lemmas, word_similarity_matrix = pre_processed_data

	if (e=='word-based'):
		clusters = w2v_word.w2v_word( word_similarity_matrix,words,selected_lemmas,s,alg)
	else:
		clusters = cluster.cluster(alg,sentence_similarity_matrix)

	L = df.ix[:,1].tolist()
	L0 = df0.ix[:,1].tolist()
	clusters_adapted = [clusters[L.index(x)] for x in L0 if x in L]
	
	Mt2 = cross1(clusters_adapted)
	Mt1 = cross1(df0.ix[:,0].tolist())

	m = 0
	try:
		from collections import Counter
		cnt = Counter(clusters_adapted)
		m = statistics.median(cnt.values())
	except:
		pass


	return m,clusters_adapted,len(set(clusters_adapted)),float(numpy.sum(abs(Mt1-Mt2))),float(numpy.sum(numpy.multiply(Mt1,Mt2)))


similarities = ['num_word_similarity','total_set_similairy','max_set_similairy','vec_similairy']
f0 = 'n50_dn_bc'
f = 'n1400_dn'
element_type = ['word-based','sentence-based']
alg = 'Spectral_Clustering_12'



total_number = len(similarities)*len(element_type)

results = []
import pandas as pd

j= 0

file0 = '/home/roozbeh/data/wiki/data/'+f0+'.csv'
file = '/home/roozbeh/data/wiki/data/'+f+'.csv'
df0 = pd.read_csv(file0)
df = pd.read_csv(file)
pre_processed_data = pre_process(df)
words,sentences,selected_lemmas, word_similarity_matrix = pre_processed_data
for s in similarities:
	sentence_similarity_matrix = w2v_sentence.calculate_matrix(sentences,s,0)
	file_utils.save_to_file(sentence_similarity_matrix,'similarity'+s+'.pckl',)
	for e in element_type:
		j = j + 1
		print(j,"/",total_number)
		print(s,e,f)
		
		
		m, cl,nn, dm,pm = evaluate(df0,df,pre_processed_data,s,e,alg)
		
		print("m: ",m)
		print(dm,pm)

		result = [j,s,nn,e,f,cl,dm,pm,m]
		results.append(result)
		cols = ['index','Similarity_Type','Number of clusters(output)', 'Word/Sentence','File_Name','Clusters','Mismatch Metric','Match Metric','M']
		panda_results = pd.DataFrame(results)
		panda_results.columns = cols
		panda_results.to_csv("/home/roozbeh/data/wiki/results/all_indexed.csv")

		df0['clusters'] = cl
		df0 = df0.sort_values(['clusters'], ascending=[True])
		df0.to_csv('/home/roozbeh/data/wiki/results/general_'+str(j)+'.csv')

		
		print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')




