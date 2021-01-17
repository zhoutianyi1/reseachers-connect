import json
import sys
from collections import defaultdict
import numpy as np
from gensim import models

from utils import tokenize, stem, cosineSimilarity
####### Reference url: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ #########

objs = []    
sentences = []  # sentences for training word2vec model

ignore = set("for the a an of for on in and to as or : , . ( ) ? !".split(" "))   # word to ignore when parsing research paper title

author2paper = defaultdict(list)   
paper2author = defaultdict(dict)


with open('trial_data.json') as f:
  data = json.load(f)
  length = len(data)
  for i, obj in enumerate(data):
    if i > 50: break
    processed = [stem(word.lower()) for word in tokenize(obj['title']) if word.lower() not in ignore and not word.isdigit()] 
    obj["processed"] = processed
    sentences.append(processed)
    objs.append(obj)

    # configure paper2author and add processed arr as one of the key val pairs
    paper2author[obj["title"]]['author'] = obj["author"]
    paper2author[obj["title"]]['processed'] = processed
    paper2author[obj["title"]]['similarity'] = [float('inf')] * length
    paper2author[obj["title"]]['index'] = i

    for author in obj["author"]:
      author2paper[author].append(obj["title"])

# train the model and get the feature vector for each paper
model = models.Word2Vec(sentences, min_count=1, size=7, window=2)
for key in paper2author:
  vector = []
  for stemmedWord in paper2author[key]['processed']:
    vector.append(model[stemmedWord])
  paper2author[key]['feature'] = vector 
print('trained done')
# get cosSim for each pair
for key1 in paper2author:
  for key2 in paper2author:
    paper1, paper2 = paper2author[key1], paper2author[key2]
    index1, index2 = paper1['index'], paper2['index']
    if index1 >= index2: continue    #  only compare when index1 < index2 to minimize comparisons
    try: 
      cosSim = cosineSimilarity(paper1['feature'], paper2['feature'])
      paper1['similarity'][index2] = paper2['similarity'][index1] = cosSim
    except:  # some edge case has research paper title with only number
      continue
print('loop done')
model.save('model.bin')
for key in paper2author:
  for i, arr in enumerate(paper2author[key]['feature']):
    paper2author[key]['feature'][i] = paper2author[key]['feature'][i].tolist()
print(paper2author)
with open('processed.json', 'w') as f:
  json.dump(paper2author, f, indent=4)

###########  Ex: paper2author:  ###########
# { 'Machine Translation Demonstration': {'author': ['Ulrike Schwall'],
#                                         'processed': ['lmt', '-', 'machin', 'translat', 'demonstr'],
#                                         'feature': [0.001, 0.2222, 0.22233, 0.192, 0.785, 0.12],
#                                         'index': 0,
#                                         'similarity': [inf, 1.28372, 5.4857, 1.38472, 0.2222]
#                                        }
#   'A Logical Operational Semantics of Full Prolog': {'author': ['Egon Börger'],
#                                                      'processed': ['logic', 'oper', 'semant', 'full', 'prolog'],
#                                                      'feature': [0.01, 0.675, -0.226, -.19, 0.785, 0.10],
#                                                      'index': 1,
#                                                      'similarity': [8.371, inf, 3.660, 5.22, 9.90]
#                                                     }
#   .....
# }

###########  Ex. author2paper  ###########
# {'Christopher Habel': ['Cognitive Linguistics: The Processing of Spatial Concepts'],
#  'Stefan Böttcher': ['Attribute Inheritance Implemented on Top of a Relational Database System']
# }



