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
ls = [' '.join([wnl.lemmatize(t).encode('utf-8') for t in ts]) for ts in tokenized_sents]


import clustercat as cc
cl = cc.cluster(in_file='input.txt', min_count=1)
