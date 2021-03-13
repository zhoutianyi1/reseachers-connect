import json
import numpy as np
from gensim import models
from utils import tokenize, stem, cosineSimilarity
####### Reference url: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ #########

objs = []    
sentences = []  # sentences for training word2vec model
paper2author = {}
paper2feature = {}

ignore = set("for the a an of for some on in and is was were are will be to as or i.e. $ { } ^ = : , . ... ( ) ? ! \ /".split(" "))   # word to ignore when parsing research paper title



with open('processed_data.json') as f:
  data = json.load(f)
  for i, obj in enumerate(data):
    if i > 100000: break
    processed = [stem(word.lower()) for word in tokenize(obj['title']) if word.lower() not in ignore and not word.isdigit()] 
    obj["processed"] = processed
    sentences.append(processed)
    objs.append(obj)
    paper2author[obj["title"]] = obj["author"]

model = models.Word2Vec(sentences, min_count=1, size=7, window=2)
model.save("model.bin")


for obj in objs:
  vector = []
  for stemmedWord in obj['processed']:
    vector.append(model[stemmedWord].tolist())
  paper2feature[obj['title']] = vector
print('trained done')

with open('paper2feature.json', 'w') as f:
  json.dump(paper2feature, f, indent=4)
