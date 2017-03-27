"""
1.   read a bunch of text files and make them into token lists
2.   calculate most similar words to one word vector
"""



contents = [];

import glob
file_list = glob.glob("genia/text/0/*.txt");
for file_name in file_list:
    with open(file_name) as f:
        content = f.readlines()
        contents.append(content)



import itertools    
all_contents = list(itertools.chain.from_iterable(contents));
print 1


from nltk.tokenize import word_tokenize
tokenized_sents = [word_tokenize(l) for l in all_contents]
print 2

import gensim
model = gensim.models.Word2Vec(tokenized_sents, size = 20, window=1, min_count=1, workers=4)

model = gensim.models.Word2Vec() # an empty model, no training
model.build_vocab(tokenized_sents)
model.train(tokenized_sents)

print model.most_similar(positive=['critical'])