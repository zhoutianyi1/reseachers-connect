import json
import numpy as np
from gensim.models import Word2Vec
from utils import tokenize, stem, cosineSimilarity

model = Word2Vec.load('model.bin')

validateTitle = "Robust adaptive single neural control for a class of uncertain nonlinear systems with input nonlinearity"
ignore = set("for the a an of for on in and to as or : , . ( ) ? !".split(" "))   # word to ignore when parsing research paper title

processed = [stem(word.lower()) for word in tokenize(validateTitle) if word.lower() not in ignore and not word.isdigit()]

def getFeatureVector(wordList):
  res = []
  for word in wordList:
    try:
      res.append(model[word])
    except:  # if the word has never been seen in the model, then append a numpy array of all zeroes
      res.append(np.zeros(7, dtype=np.float32))  # 7 is the model word2vec features hyperparameter
  return res

feature = getFeatureVector(processed)

author = ['Bryant Zhou']
recommended = []

with open('processed.json') as f:
  paper2author = json.load(f)
  # print(paper2author)
  length = len(paper2author)
  newIdx = length
  paper2author[validateTitle] = {}
  paper2author[validateTitle]['author'] = author
  paper2author[validateTitle]['feature'] = feature
  paper2author[validateTitle]['processed'] = processed
  paper2author[validateTitle]['similarity'] = [float('inf')] * (length + 1)
  paper2author[validateTitle]['index'] = newIdx

  for key in paper2author:
    print(paper2author[key]['feature'])
    print(f'new: {feature}')
    sim = cosineSimilarity(paper2author[key]['feature'], feature)
    paper2author[key]['similarity'].append(sim)
    idx = paper2author[key]['index']
    paper2author[validateTitle]['similarity'][idx] = sim
    if sim < 1:
      recommended.append((key, paper2author[key]['author']))
  
print(recommended)