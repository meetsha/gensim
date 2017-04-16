import logging
import os
from pprint import pprint
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
    lsi = models.LsiModel.load('tmp/model.lsi')
    lda = models.LdaModel.load('tmp/model.lda')    

new_doc = open('sample.txt', 'r', encoding = 'utf-8').read()
print(new_doc)

new_vec = dictionary.doc2bow(new_doc.lower().split())
pprint(new_vec)

vec_tfidf = tfidf[new_vec]
pprint(vec_tfidf)

vec_lsi = lsi[new_vec]
pprint(vec_lsi)

vec_lda = lda[new_vec]
pprint(vec_lda)
