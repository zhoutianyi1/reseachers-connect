from sentence_transformers import SentenceTransformer
import numpy as np
import hnswlib
import sys
import json
import time

model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
target = "Asynchronous Consensus with Bounded Memory"
targetVector = model.encode(target)

start = time.time()
sentences = []
with open('trial_data.json', 'r') as f:
    data = json.load(f)
    for i, obj in enumerate(data):
        sentences += [obj[str(i)]['title']]
end = time.time()
print(f'loading json takes {end - start} seconds')
p = hnswlib.Index(space='cosine', dim=300)

start = time.time()
p.load_index("title.bin", max_elements = 8000000)
middle = time.time()
labels, distance = p.knn_query(targetVector, k=10)
end = time.time()


for labelIndex in labels[0]:
    print(sentences[labelIndex])

print(f'loading title.bin takes {middle-start} seconds and looking up takes {end - middle} seconds')
