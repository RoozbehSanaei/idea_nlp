import utils
import numpy

f = 'n50_de_rma'
file = '/home/roozbeh/data/wiki/data/'+f+'.csv'

similarities = ['num_word_similarity','total_set_similairy','max_set_similairy','vec_similairy','skipthoughts_similarity']
similarities = ['skipthoughts_similarity']


sents = utils.extract_col(file,2)

for s in similarities:
	M = utils.load_from_file("similarity"+s+".pickl")
	utils.xls_with_header(sents,M,s+".xls")