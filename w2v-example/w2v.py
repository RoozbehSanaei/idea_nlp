""" 
Load a model (google model here)
find most similar words to a particular words
"""
import gensim
model = gensim.models.Word2Vec.load_word2vec_format('C:\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin', binary=True);
model.most_similar(positive=['sustain'])