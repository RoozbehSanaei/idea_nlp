
contents = [];

import numpy

def vec(doc):
    x = numpy.zeros(100)
    for e in doc:
        x[e[0]]=e[1]
    return x
        


import glob
file_list = glob.glob("*.txt");
for file_name in file_list:
    with open(file_name) as f:
        content = [line.decode('latin-1').strip('"') for line in f.readlines()]
        contents.append(content)
import itertools    
all_contents = list(itertools.chain.from_iterable(contents));

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    

# compile sample documents into a list
doc_set = all_contents

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary, passes=20)

# generate topics
topics = ldamodel.print_topics(num_topics=100);

# generate a vector for each setence
vectors = [vec(ldamodel[dictionary.doc2bow(text)]) for text in texts];

# find mininimum distance between a new sentence and previous sentence
distances = [numpy.min(numpy.array([numpy.linalg.norm(vectors[j] - vectors[i]) for j in range(0,i)])) for i in range(1,2959)]

t_numbers = [0.1,0.2,0.3,0.4,0.45,0.5,0.6,0.7,0.8,0.9]

nni = []
for t in t_numbers:
	number_of_new_ideas = []
	new_ideas = []
	n = 0;
	for i in range(0,2958):
		if (distances[i]>t):
			new_ideas.append(i)
			n = n + 1
		number_of_new_ideas.append(n)
	nni.append(number_of_new_ideas)	

# plot the number of new ideas vs number of idea
import matplotlib.pyplot as plt
for i in range(0,9):
	plt.plot(nni[i])
plt.show()

for i in range(1,100):
	 print(all_contents[new_ideas[-i]])