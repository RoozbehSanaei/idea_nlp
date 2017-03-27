import csv
import gensim
import numpy
import networkx as nx
from tqdm import tqdm



print("loading file...",end=" ",flush=True)
with open('superset.csv') as input_file:
	rows = csv.reader(input_file, delimiter=';')
	res = list(zip(*rows))
print("done!",flush=True)


print("loading word2vec model...",end=" ",flush=True)
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")    
print("done!",flush=True)

words = [(w.split(' '))[0] for w in res[0] if (w.split(' '))[0] in model.wv.vocab]
words = list(set(words))

tqdm(desc="Make Similarity Matrix")
M = [[model.similarity(w1, w2) for w1 in words] for w2 in tqdm(words)]

for i in range(len(words)):
	for j in range(len(words)):
		if (M[i][j]<=0.5):
			M[i][j]=0
		else:
			M[i][j]=1


print("create networkx graph...",end=" ",flush=True)
G=nx.from_numpy_matrix(numpy.array(M))
print("done!",flush=True)

print("compute clusters...",end = " ",flush=True)
gen = nx.k_clique_communities(G, 4)
c = list(gen)
print(list(c[0]))
print("done!",flush=True)


L = [[words[i],labels[i]] for i in range(len(labels))]

with open("clusters0.csv", "w") as output_file:
	writer = csv.writer(output_file, delimiter=';')
	writer.writerows(L)



