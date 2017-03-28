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

#model = gensim.models.Word2Vec.load_word2vec_format('C:\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin', binary=True);

print model.most_similar(positive=['critical'])