import numpy
from multiprocessing.pool import ThreadPool
import random
import math
import pre_processing
import pandas as pd
import similarity_functions
import file_utils
import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models



similarities = ['total_set_similairy','skipthoughts_similarity','skipthoughts_similarity_N','num_word_similarity','max_set_similairy','vec_similairy'];
#similarities = ['vec_similairy'];

#pre-processing and extracting words
f = 'n1400_dn'
file = '/home/roozbeh/data/wiki/data/'+f+'.csv'
df = pd.read_csv(file)
words,sentences,selected_lemmas, word_similarity_matrix = pre_processing.pre_process(df)


#skipthoughts model
l = len(sentences)
import skipthoughts
sentence_model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(sentence_model)


pool = ThreadPool(processes=22)

#gensim word2vec model
from gensim import corpora, models
global model 
model = gensim.models.Word2Vec.load("wiki_files/wiki.en.word2vec.model")    


inputs = [0 for i in range(l*l)]
sentences_strs = [" ".join(sent) for sent in sentences]
sentence_vectors = encoder.encode(sentences_strs)
sentence_word_indices = [[words.index(w) for w in sent if w in words] for sent in sentences]
total_vector = [sum([model.wv[w] for w in sent if w in model.wv.vocab]) for sent in sentences]



#gensim lda model
tokenizer = RegexpTokenizer(r'\w+')
en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()
texts = []

for i in sentences_strs:
	raw = i.lower()
	tokens = tokenizer.tokenize(raw)
	stopped_tokens = [i for i in tokens if not i in en_stop]
	stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	texts.append(stemmed_tokens)


dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=200, id2word = dictionary, passes=10)
LDA_vectors = [[l[1] for l in ldamodel[dictionary.doc2bow(text)]] for text in texts]



def total_set_similairy(A,B):
    s = 0
    for i in A:
        for j in B:
        	s = s + word_similarity_matrix[i][j]
    s = s / (len(A)*len(B)+1)
    return s


def max_set_similairy(A,B):
    m = 0
    for i in A:
        for j in B:
                if (word_similarity_matrix[i][j]>m):
                    m = word_similarity_matrix[i][j]
    return m



def vec_similairy(i,j):
    p = 0
    p = numpy.dot(total_vector[i],total_vector[j])/(numpy.linalg.norm(total_vector[i])*numpy.linalg.norm(total_vector[j])+0.000001)
    return p


def num_word_similarity(A,B):
    l = len(set(A) & set(B))
    return l


from tqdm import trange
k = 0;
for i in trange(l):
	for j in range(l):
		inputs[k] = [i,j]
		k = k + 1;



mat = [[0 for i in range(l)] for j in range(l)]

global counter; counter = 0 

def f(x,s):
	global counter
	i = x[0]
	j = x[1]

	if (s=='num_word_similarity'):
		mat[i][j] = similarity_functions.num_word_similarity(sentences[i],sentences[j])
	elif (s=='total_set_similairy'):
		mat[i][j] = total_set_similairy(sentence_word_indices[i],sentence_word_indices[j])
	elif (s=='max_set_similairy'):
		mat[i][j] = max_set_similairy(sentence_word_indices[i],sentence_word_indices[j])
	elif (s=='vec_similairy'):
		mat[i][j] = vec_similairy(i,j)
	elif (s=='skipthoughts_similarity'):
		mat[i][j] = numpy.dot(sentence_vectors[i],sentence_vectors[j])
	elif (s=='skipthoughts_similarity_N'):
		mat[i][j] = numpy.dot(sentence_vectors[i],sentence_vectors[j])/(numpy.linalg.norm(sentence_vectors[i])*numpy.linalg.norm(sentence_vectors[j])+0.000001)
	

	counter = counter + 1
	print(s,counter)





def parallel_proc(f,inputs,s):
	
	number_of_threads=20
	l = len(inputs)
	m = math.ceil(l/number_of_threads)

	def ff(l):
		return [f(x,s) for x in l]

	def ind(i):
		if (i<number_of_threads):
			return i*m
		elif (i==number_of_threads):
			return l

	async_result = [pool.apply_async(ff, (inputs[ind(i):ind(i+1)],)) for i in range(number_of_threads)]

	return_vals = [async_result[i].get() for i in range(number_of_threads)]

	results=numpy.concatenate(return_vals, axis=0);
	return results


for s in similarities:
	parallel_proc(f,inputs,s)
	file_utils.save_to_file(mat,s+".pickl")
	#file_utils.xls_without_header(mat,s+".csv")
	counter = 0 

s = 'vec_similairy'
mat = file_utils.load_from_file(s+".pickl")
ind = 1
s = numpy.argsort(mat[1])[::-1]