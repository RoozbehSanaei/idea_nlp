
from gensim import corpora, models

## jj indicates index of file used 
jj = 3


#load csv file 
import csv
files = ['n1400 distributed novice.csv','n1400 group experienced.csv','n100 group novice.csv','n100 distributed experienced.csv']
with open(files[jj]) as input_file:
	rows = csv.reader(input_file, delimiter=',')
	res = list(zip(*rows))

doc_set =  [l.strip('"') for l in res[1]]



from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
from stop_words import get_stop_words
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()
    


# list for tokenized documents in loop
words = []


from nltk.stem import WordNetLemmatizer
w_lem = WordNetLemmatizer()

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # lemmatize tokens
    lemmatized_tokens = [w_lem.lemmatize(i) for i in stopped_tokens]
    
    # add tokens to list
    words = words+lemmatized_tokens


#load word2vec ,model
import gensim
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")    

# remove repetitive words or those that are not found in wikipedia vocabulary
words = [w for w in words if w in model.wv.vocab]
words = list(set(words))

#make the similarity model
M = [[model.similarity(w1, w2) for w1 in words] for w2 in tqdm(words)]


#ignore negative corrolation
import numpy

for i in range(len(words)):
	for j in range(len(words)):
		if (M[i][j]<0):
			M[i][j]=0

#cluster
import numpy
from sklearn.cluster import AffinityPropagation
af = AffinityPropagation().fit(M)
labels = af.labels_

# sum of similarities for each word
S = numpy.sum(M,axis=1)


n = len(words)

V = numpy.zeros((max(labels)+1,400))
N = numpy.zeros(400)
W = [0]*n
for i in range(n):
    W[i] = model.wv[words[i]]
    V[labels[i]] = V[labels[i]] + W[i]

for i in range(max(labels)+1):
    V[i] = V[i] / numpy.linalg.norm(V[i])

max_similarity_word = [0]*(max(labels)+1)
max_dot_product_word = [0]*(max(labels)+1)
for i in range(max(labels)+1):
    cluster_indices = numpy.where(labels==i)[0];
    cluster_words = [words[l] for l in cluster_indices]
    cluster_words_total_similarity = [S[l] for l in cluster_indices]
    cluster_words_dot_products = [numpy.dot(V[i],W[l]) for l in cluster_indices]
    max_similarity_word[i] = cluster_words[cluster_words_total_similarity.index(max(cluster_words_total_similarity))]
    max_dot_product_word[i] = cluster_words[cluster_words_dot_products.index(max(cluster_words_dot_products))]
    print(cluster_words)
    print(max_similarity_word[i],max_dot_product_word[i])

#create, sort, and save the clusters
L1 = [[words[i],labels[i],S[i],max_similarity_word[labels[i]],max_dot_product_word[labels[i]]] for i in range(len(labels))]


L1.sort(key=lambda x: x[1])
with open("clusters"+str(jj+1)+".csv", "w") as output_file:
	writer = csv.writer(output_file, delimiter=';')
	writer.writerows(L1)