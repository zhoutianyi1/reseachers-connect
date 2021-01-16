import json
import numpy as np
import sys
from utils import tokenize, stem
from collections import defaultdict
from gensim import models

objs = []

ignore = set("for the a an of for on in and to as or : , . ( ) ? !".split(" "))

author2paper = defaultdict(list)
paper2author = defaultdict(dict)


with open('trial_data.json') as f:
  data = json.load(f)
  for obj in data:
    processed = [stem(word.lower()) for word in tokenize(obj['title']) if word.lower() not in ignore and not word.isdigit()] 
    obj["processed"] = processed
    objs.append(obj)

    paper2author[obj["title"]]['author'] = obj["author"]
    paper2author[obj["title"]]['processed'] = processed

    for author in obj["author"]:
      author2paper[author].append(obj["title"])
############  Ex: paper2author:
# { 'Machine Translation Demonstration': {'author': ['Ulrike Schwall'],
#                                         'processed': ['lmt', '-', 'machin', 'translat', 'demonstr']
#                                        }
#   'A Logical Operational Semantics of Full Prolog': {'author': ['Egon Börger'],
#                                                      'processed': ['logic', 'oper', 'semant', 'full', 'prolog']
#                                                     }
# }

############  Ex. author2paper
# {'Christopher Habel': ['Cognitive Linguistics: The Processing of Spatial Concepts'],
#  'Stefan Böttcher': ['Attribute Inheritance Implemented on Top of a Relational Database System']
# }
print(paper2author)
# sys.exit()
sentences = []
for key in paper2author:
  sentences.append(paper2author[key]['processed'])

model = models.Word2Vec(sentences, min_count=1, size=7, window=2)

for key in paper2author:
  vector = []
  for stemmedWord in paper2author[key]['processed']:
    vector.append(model[stemmedWord])
  paper2author[key]['feature'] = vector 


# train model
# model = models.Word2Vec(sentences, min_count=1, size=10, window=2)
# # summarize the loaded model
# words = list(model.wv.vocab)
# print(words)
# print(model['second'])




# print(sentences)
# print(objs)