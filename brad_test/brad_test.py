import file_utils
import graph_utils
import numpy

f = 'n50_de_rma'
file = '/home/roozbeh/data/wiki/data/'+f+'.csv'

similarities = ['num_word_similarity','total_set_similairy','max_set_similairy','vec_similairy','skipthoughts_similarity','skipthoughts_similarity1']

similarities = ['skipthoughts_similarity']

threshold = 1.1

s = similarities[0]
sents = file_utils.extract_col(file,2)
M = file_utils.load_from_file("similarity"+s+".pickl")
file_utils.xls_with_header(sents,M,s+".xls")
M[M<threshold] = 0
M[M>=threshold] = 1
import networkx as nx
G=nx.from_numpy_matrix(M)
cc = sorted(nx.connected_components(G), key = len, reverse=True)

cl = graph_utils.cluster_rep(cc)



file_utils.lists_to_csv([sents,cl],['sent','cluster'],'results.csv')