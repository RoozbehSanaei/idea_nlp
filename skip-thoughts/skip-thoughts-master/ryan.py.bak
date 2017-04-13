contents = [];

import glob
file_list = glob.glob("*.txt");
for file_name in file_list:
    with open(file_name) as f:
        content = [line.decode('latin-1').strip('"') for line in f.readlines()]
        contents.append(content)
import itertools    
all_contents = list(itertools.chain.from_iterable(contents));

from nltk.stem import WordNetLemmatizer
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

w_lem = WordNetLemmatizer()
# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    #stemmed_tokens = [p_stemmer.stem(i) for i in tokens]

    # lemmitization
    #stemmed_tokens = [w_lem.lemmatize(i) for i in tokens]
    
    # add tokens to list
    texts.append(stopped_tokens)

sentences = [' '.join(text) for text in texts if text != []]

import skipthoughts
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
vectors = encoder.encode(sentences)
import numpy as np
Nv1v2 = [[np.dot(v1,v2) for v1 in vectors] for v2 in vectors]
S = np.argsort(Nv1v2[0])
