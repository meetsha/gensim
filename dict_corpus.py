import logging
import os
from gensim import corpora
from collections import defaultdict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = []

for name in os.listdir('documents'):
    if name.endswith('.txt'):
        try:
            print(name)
            doc = open('documents/' + name).read()
            # print(doc)
            documents.append(doc)
        except:
            pass


# documents.append(open("sample.txt", encoding='utf-8').read())
# print(documents)

stopfile = open('stopwords.txt')
stoplist = []
for line in stopfile.readlines():
    line = line.rstrip('\n')
    stoplist.append(line)

# print(stoplist)

texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]

# print(texts)

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 2]
         for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.save('tmp/mesh.dict')

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('tmp/corpus.mm', corpus)  # store to disk, for later use
