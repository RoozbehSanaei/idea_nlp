""" 
Load a model (google model here)
find most similar words to a particular words
"""
import csv
import gensim
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")    
model = gensim.models.KeyedVectors.load_word2vec_format('/home/roozbeh/data/wiki/GoogleNews-vectors-negative300.bin', binary=True);
print model.most_similar(positive=['road'])
print model.similarity("road", "bicycle")

with open('superset.csv') as input_file:
	rows = csv.reader(input_file, delimiter=';')
	res = list(zip(*rows))
