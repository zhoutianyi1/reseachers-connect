import json
import numpy as np
from gensim.models import Word2Vec
from utils import tokenize, stem, cosineSimilarity
import time
model = Word2Vec.load('model.bin')

validateTitle = "Robust adaptive single neural control for a class of uncertain nonlinear systems with input nonlinearity"
# validateTitle = "An Agent-based Architecture for Analyzing Business Processes of Real-Time Enterprises"
# validateTitle = "ROFL: Routing on Flat Labels"
# validateTitle = "Private Anomaly Detection Across ISP Networks"
ignore = set("for the a an of for on in and to as or : , . ( ) ? !".split(" "))   # word to ignore when parsing research paper title
processed = [stem(word.lower()) for word in tokenize(validateTitle) if word.lower() not in ignore and not word.isdigit()]

def getFeatureVector(wordList):
  res = []
  for word in wordList:
    try:
      res.append(model[word])
    except:  # if the word has never been seen in the model, then append a numpy array of all zeroes
      res.append(np.zeros(7, dtype=np.float32))  # 7 is the hard-coded model word2vec features hyperparameter
  return res

feature = getFeatureVector(processed)

author = ['Bryant Zhou']
recommended = []

start = time.time()
with open('paper2feature.json') as f:
  objs = json.load(f)
  middle = time.time()
  print(f'loading takes {middle - start} seconds')

  for title in objs:
    if len(objs[title]) <= 2: continue
    try:
      sim = cosineSimilarity(objs[title], feature)
      # print(sim)
    except:
      sim = 0.1 
      
    if sim > 0.9:
      # print(sim)
      recommended.append(title)
  end = time.time()
  print(f'looping 100k takes {end - middle} secs')
print(recommended)
print(len(recommended))
