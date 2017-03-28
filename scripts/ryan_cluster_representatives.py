import csv
with open('ryan.csv') as input_file:
	rows = csv.reader(input_file, delimiter=',')
	res = list(zip(*rows))

import numpy
words = [(res[0][i].split(' '))[0] for i in range(1,len(res[0]))]
POS_TAGS = [(res[0][i].split(' '))[1] for i in range(1,len(res[0]))]
labels = numpy.array([int(res[1][i]) for i in range(1,len(res[0]))])


import gensim
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")   





#make the similarity model
M = [[model.similarity(w1, w2) for w1 in words] for w2 in words]


#ignore negative corrolation

for i in range(len(words)):
	for j in range(len(words)):
		if (M[i][j]<0):
			M[i][j]=0


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


res = res + [0] + [0]
#create, sort, and save the clusters
res[7] = ['word with maximum total similairity']+[max_similarity_word[labels[i]] for i in range(len(labels))]
res[8] = ['word with maximum total dot products']+[max_dot_product_word[labels[i]] for i in range(len(labels))]


res1 = list(map(list, zip(*res)))

with open("clusters_ryan.csv", "w") as output_file:
	writer = csv.writer(output_file, delimiter=';')
	writer.writerows(res1)
