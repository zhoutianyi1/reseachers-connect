import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CBOW(nn.Module):
  def __init__(self, size, embeddingFeatures, contextSize):
    super(CBOW, self).__init__()
    self.size = size
    self.embeddingFeatures = embeddingFeatures
    self.contextSize = contextSize

    self.embedding = nn.Embedding(self.size, self.embeddingFeatures)
    self.dense1 = nn.Linear(self.contextSize * self.embeddingFeatures, 256)
    self.dense2 = nn.Linear(256, size)

  def forward(self, input):
    out = self.embedding(input).view(1, -1)
    out = F.relu(self.dense1(out))
    out = self.dense2(out)
    out = F.log_softmax(out, dim=1)
    return out

  def getEmbeddingMatrix(self, input):
    return self.embedding(input)

  

