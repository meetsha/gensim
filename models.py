import logging
import os
from gensim import corpora, models, similarities
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if (os.path.exists("tmp/mesh.dict")):
   dictionary = corpora.Dictionary.load('tmp/mesh.dict')
   corpus = corpora.MmCorpus('tmp/corpus.mm')
   print("Used files generated from dict_corpus")
else:
   print("Please run dict_corpus.py to generate data set")

if (os.path.exists("tmp/model.tfidf")):
    tfidf = models.TfidfModel.load('tmp/model.tfidf')
else:
    tfidf = models.TfidfModel(corpus)
    tfidf.save('tmp/model.tfidf')
corpus_tfidf = tfidf[corpus]

if (os.path.exists("tmp/model.lsi")):
    lsi = models.LsiModel.load('tmp/model.lsi')
else:
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    lsi.save('tmp/model.lsi')
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(10)

if (os.path.exists("tmp/model.lda")):
    lda = models.LdaModel.load('tmp/model.lda')
else: 
    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=200, update_every=0, passes=10)
    lda.save('tmp/model.lda')
corpus_lda = lda[corpus_tfidf]
lda.print_topics(10)
