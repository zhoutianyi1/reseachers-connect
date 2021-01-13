import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import CBOW


#### Training data initialize ####
researchName = "Fast End-to-End Embedding Learning for Video Object Segmentation".split()
researchName = list(map(lambda x: x.lower(), researchName))
trainingDataRaw = []
for i in range(2, len(researchName) - 2):
  context = [researchName[i-2], researchName[i-1], researchName[i+1], researchName[i+2]]
  target = researchName[i]
  trainingDataRaw.append((context, target))


#### lookup table for vector ####
vocabs = set(researchName)
wordToIndex = { word: i for i, word in enumerate(vocabs) }


#### Model Initialization ####
criterion = nn.NLLLoss()
model = CBOW(len(vocabs), 5, 4)
optimizer = optim.Adam(model.parameters(), lr = 0.01)


#### Training loop ####
for epoch in range(3):
  for context, target in trainingDataRaw:
    contextIdx = torch.tensor([wordToIndex[word] for word in context], dtype=torch.long)
    model.zero_grad()
    output = model(contextIdx)

    loss = criterion(output, torch.tensor([wordToIndex[target]], dtype=torch.long))
    loss.backward()
    optimizer.step()
    print(model.getEmbeddingMatrix(contextIdx))


