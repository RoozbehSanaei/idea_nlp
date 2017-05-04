import numpy
import gensim

from gensim import corpora, models
global model 
model = gensim.models.Word2Vec.load("wiki_files/wiki.en.word2vec.model")    



def total_set_similairy(A,B):
    global model 
    s = 0
    for w1 in A:
        for w2 in B:
            if (w1 in model.wv.vocab) & (w2 in model.wv.vocab):
                s = s + model.similarity(w1,w2)
    s = s / (len(A)*len(B)+1)
    return s

def max_set_similairy(A,B):
    m = 0
    for w1 in A:
        for w2 in B:
            if ((w1 in model.wv.vocab) & (w2 in model.wv.vocab)):
                if (model.similarity(w1,w2)>m):
                    m = model.similarity(w1,w2)
    return m



def vec_similairy(A,B):
    p = 0
    w_A = sum([model.wv[w] for w in A if w in model.wv.vocab])
    w_B = sum([model.wv[w] for w in B if w in model.wv.vocab])
    if ((numpy.linalg.norm(w_A)!=0) & (numpy.linalg.norm(w_B)!=0)):
        p = numpy.dot(w_A,w_B)/(numpy.linalg.norm(w_A)*numpy.linalg.norm(w_B)+0.000001)
    return p



def skipthoughts_similarity_N(A,B):
    SA = " ".join(A)
    SB = " ".join(B)
    vectors = encoder.encode([SA,SB])
    r = numpy.dot(vectors[0],vectors[1])/(numpy.linalg.norm(vectors[0])*numpy.linalg.norm(vectors[1])+0.000001)
    print(r)
    return r


def skipthoughts_similarity(A,B):
    SA = " ".join(A)
    SB = " ".join(B)
    vectors = encoder.encode([SA,SB])
    print(vectors)
    r = numpy.dot(vectors[0],vectors[1])
    print(r)
    return r



def num_word_similarity(A,B):
    l = len(set(A) & set(B))
    return l

