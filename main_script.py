import w2v_word_clusters
import pandas as pd
from tqdm import tqdm
import pre_processing
import similarity_functions
import numpy
import k_clique_cluster

files = [
'n100_de','n100_dn_v3','n100_gn','n50_de_rma','n50_gn_bc',
'n100_de_v3','n100_ge','n100_gn_v3','n50_ge_bc','n50_gn_rma',
'n100_dn','n100_ge_v3','n50_de_bc','n50_ge_rma'
]


#load file
f = 'n100_de'
i = 2 #similarity metric
j = 0 #word/sentence
k = 0 #w2v+affinity/co-occurance+k-clique

file = '/home/roozbeh/data/wiki/data/'+f+'.csv'


df = pd.read_csv(file)


tqdm.pandas(desc="Make Spacy Tokens")
df['tokens'] = df.ix[:,2].progress_apply(lambda x: pre_processing.cleanPassage(x))    
df['lemmas'] = df['tokens'].apply(lambda x: pre_processing.getLemmas(x))
sentences = list(df['lemmas'])
probs_cutoff_lower = pre_processing.findMeaningfulCutoffProbability([t for tok in df['tokens'] for t in tok])
tqdm.pandas(desc="Remove redundant lemmas")
selected_lemmas = df['tokens'].progress_apply(lambda x: pre_processing.makeNodelist(x,probs_cutoff_lower))


if (j==0):
	print("cluster based on words")

	#if (k==0):
	cluster_words = w2v_word_clusters.w2v_word_clusters(selected_lemmas)
	#else
	#cluster_words = k_clique_cluster.get_cluster(f,4)


	if (i==1):
		M = [[similarity_functions.num_word_similarity(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
	elif (i==2):
		M = [[similarity_functions.total_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
	elif (i==3):
		M = [[similarity_functions.max_set_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]
	elif (i==4):
		M = [[similarity_functions.vec_similairy(cw,sl) for cw in cluster_words ] for sl in selected_lemmas]

	clusters =  [M[i].index(max(M[i])) for i in range(len(M))]

else:
	print("cluster based on sentences")
	import w2v_sentence_clusters
	clusters = w2v_sentence_clusters.w2v_sentence_clusters(sentences,i)




result_file = open('results/'+f+'_'+str(i)+'_'+str(j)+'.txt', "w")
print(clusters)
print("number of clusters : ", len(clusters))


#jaccard evaulaton
import jaccard                

data_df = [[str(df.ix[:,0][i]),str(df.ix[:,1][i])] for i in range(len(clusters))]
data_cl = [[str(clusters[i]),str(df.ix[:,1][i])] for i in range(len(clusters))]
d1 = pre_processing.david_dict(data_df)
d2 = pre_processing.david_dict(data_cl)
jaccard.greedy_match_sets(d1, d2,result_file,len(selected_lemmas))
result_file.close()


#save to file
L1 = [[clusters[i],df.ix[:,1][i]] for i in range(len(selected_lemmas))]
L1.sort(key=lambda x: x[0])
results = pd.DataFrame(L1)
results.to_csv("clusters_w2v_h")
