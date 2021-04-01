from sentence_transformers import SentenceTransformer
import numpy as np
import hnswlib
import sys
import json
import time

###### Get Embeddings of sentences ######
model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
sentences = []
with open('trial_data.json', 'r') as f:
    data = json.load(f)
    for i, obj in enumerate(data):
        sentences += [obj[str(i)]['title']]

sentences_embeddings = model.encode(sentences)
########################################


###### hnswlib ######
dim = 300   # 300 is from the model
num_elements = 8000000
data = sentences_embeddings
labels = sentences
p = hnswlib.Index(space='cosine', dim = dim)
p.init_index(max_elements = num_elements, ef_construction=200, M = 50)


start = time.time()
for i in range(len(data)):
    # print(data[i], labels[i])
    p.add_items(data[i], i)
end = time.time()

print(f'adding items take {end-start} seconds')   # 744 seconds for 1M entry
p.set_ef(50)
p.save_index("title.bin")
del p



# target = 'Realtime multi-person 2D pose estimation is a key component in enabling machines to have an understanding of people images and videos.\
# In this work, we present a realtime approach to detect the 2D pose of multiple people in an image. The proposed \
# method uses a nonparametric representation, which we refer to as Part Affinity Fields (PAFs), to learn to associate body parts with \
# individuals in the image. This bottom-up system achieves high accuracy and realtime performance, regardless of the number of people \
# in the image. In previous work, PAFs and body part location estimation were refined simultaneously across training stages. We \
# demonstrate that a PAF-only refinement rather than both PAF and body part location refinement results in a substantial increase in both \
# runtime performance and accuracy. We also present the first combined body and foot keypoint detector, based on an internal annotated \
# foot dataset that we have publicly released. We show that the combined detector not only reduces the inference time compared to \
# running them sequentially, but also maintains the accuracy of each component individually. This work has culminated in the release of'

# targetVector = model.encode(target)
# start = time.time()
# target = 'Modeling and analysis of multistage failures of a system'
# targetVector = model.encode(target)
# labels, distances = p.knn_query(targetVector, k = 10)
# end = time.time()

# for label in labels[0]:
#     print(sentences[label])
# print(f'duration is {end - start} seconds')

