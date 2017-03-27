import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
id2word = gensim.corpora.Dictionary.load_from_text('_wordids.txt.bz2')
mm = gensim.corpora.MmCorpus('_tfidf.mm')
print(mm)
#lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)

