import gensim
from gensim import corpora, models
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")    
print(model.similarity('hello', 'hi'))
print('hi' in model.wv.vocab)
print(model.wv['hi'])