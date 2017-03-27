contents = [];

with open("output.txt") as f:
    content = f.readlines()
    contents.append(content)

lines = [line.rstrip('\n') for line in open("output.txt")]


import itertools    
all_contents = list(itertools.chain.from_iterable(contents));


from nltk.tokenize import word_tokenize
tokenized_sents = [word_tokenize(l) for l in all_contents]
print 2

import gensim
model = gensim.models.Word2Vec.load_word2vec_format('C:\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin', binary=True);

model.most_similar(positive=['result'])

import numpy
X = numpy.zeros((10574,300));
words = [None]*10574;
print "---------------"
i = 0;
j = 0;
while (i<(len(lines))):
    if lines[i] in model:
        print i
        X[j] = model[lines[i]];
        words[j] = lines[i]
        j = j + 1
    i = i + 1

numpy.savetxt("foo.csv", X,fmt = ('%1.12f'), delimiter=",")


f = open('words1.txt', 'w')        
for i in range(len(words)):
  f.write("%s\n" % words[i])
f.close()