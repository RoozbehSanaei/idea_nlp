import csv
with open('superset.csv') as input_file:
	rows = csv.reader(input_file, delimiter=';')
	res = list(zip(*rows))

import gensim
model = gensim.models.Word2Vec.load("wiki.en.word2vec.model")    

words = [(w.split(' '))[0] for w in res[0] if (w.split(' '))[0] in model.wv.vocab]
words = list(set(words))


M = [[model.similarity(w1, w2) for w1 in words] for w2 in words]

for i in range(len(words)):
	for j in range(len(words)):
		if (M[i][j]<0):
			M[i][j]=0

from sklearn.cluster import AffinityPropagation

import numpy
prod_avg = []
prod_max = []
for n in range(100,len(words)):
	print(n)
	P = numpy.array(M)[0:n,0:n]
	af = AffinityPropagation().fit(P)
	labels = af.labels_
	L = [[words[i],labels[i]] for i in range(n)]

	V = numpy.zeros((max(labels)+1,400))
	N = numpy.zeros(400)

	for i in range(n):
		V[labels[i]] = V[labels[i]] + model.wv[words[i]]
		N[labels[i]] = N[labels[i]] + 1

	for i in range(max(labels)+1):
		V[i] = V[i] / N[i]

	W = model.wv[words[n]]
	prod_avg.append(numpy.mean([abs(numpy.dot(W,V[i])) for i in range(max(labels)+1)]))
	prod_max.append(numpy.max([abs(numpy.dot(W,V[i])) for i in range(max(labels)+1)]))

import pickle
output_file = open('prod_data.pkl', 'wb')
pickle.dump([prod_avg,prod_max], output_file)
output_file.close()

import matplotlib.pyplot
matplotlib.pyplot.scatter(prod_max,prod_avg)
matplotlib.pyplot.show()

new_count = []
n = n + 1
for i in range(len(prod_max)):
	if prod_max[i]<0.15:
		n = n + 1
	new_count.append(n)	
matplotlib.pyplot.plot(new_count)
matplotlib.pyplot.show()



all_labels = []
P = numpy.array(M)
import numpy
for d in range(5,10):
	for i in range(6,16):
		print(i,d)
		af = AffinityPropagation(damping=float(d/10),preference=-10*i).fit(P)
		all_labels.append(af.labels_)


ICs = [0]*len(all_labels)
ECs = [0]*len(all_labels)
MCs = [0]*len(all_labels)
MDs = [0]*len(all_labels)
MAXs = [0]*len(all_labels)
MINs = [0]*len(all_labels)
for k in range(len(all_labels)):
	print(k)
	labels=all_labels[k]
	IC = 0
	EC = 0
	CS = [labels.tolist().count(i) for i in range(max(labels.tolist())+1)]
	for i in range(len(M)):
		for j in range(len(M)):
			if (labels[i]==labels[j]):
				IC = IC + M[i][j]/ (CS[labels[i]])
			else:
				EC = EC + M[i][j]/len(M)
	ICs[k]=IC
	ECs[k]=EC
	MCs[k]=float(sum(CS))/(max(labels.tolist())+1)
	MDs[k]= numpy.median(CS)
	MAXs[k]=max(CS)
	MINs[k]=min(CS)



import matplotlib.pyplot
matplotlib.pyplot.scatter(ICs, MDs)
matplotlib.pyplot.ylabel('cluster size median')
matplotlib.pyplot.xlabel('intra-cluster cost')
matplotlib.pyplot.show()


import pickle
output_file = open('all_labels2.pkl', 'wb')
pickle.dump([all_labels,ICs,ECs,MCs,MDs], output_file)
output_file.close()

import pickle
input_file = open('all_labels2.pkl', 'rb')
[all_labels,ICs,ECs,MCs,MDs] = pickle.load(input_file)
input_file.close()


with open("clusters0.csv", "wb") as output_file:
	writer = csv.writer(output_file, delimiter=';')
	writer.writerows(L)