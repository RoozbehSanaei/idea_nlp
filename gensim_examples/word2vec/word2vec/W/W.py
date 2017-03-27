"""
1.   read a bunch of text files and make them into token lists
2.   calculate most similar words to one word vector
"""



contents = [];

import glob
file_list = glob.glob("*.txt");
for file_name in file_list:
    with open(file_name) as f:
        content = [line.decode('latin-1').strip('"') for line in f.readlines()]
        contents.append(content)



import itertools    
all_contents = list(itertools.chain.from_iterable(contents));
print 1


from nltk.tokenize import word_tokenize
tokenized_sents = [word_tokenize(l) for l in all_contents]
print 2


import nltk
wnl = nltk.WordNetLemmatizer()
ls = [[wnl.lemmatize(t).encode('utf-8') for t in ts] for ts in tokenized_sents]

import csv
with open("output", "wb") as output_file:
	writer = csv.writer(output_file, delimiter=' ')
	writer.writerows(ls)


import gensim
model = gensim.models.Word2Vec(ls, size = 5, window=10, min_count=2, workers=4)

model = gensim.models.Word2Vec() # an empty model, no training
model.build_vocab(tokenized_sents)
model.train(tokenized_sents)

print model.most_similar(positive=['metal'])