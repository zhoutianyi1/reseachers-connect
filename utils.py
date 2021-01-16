import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from numpy import linalg as LA

stemmer = SnowballStemmer('english')

def tokenize(sentence): return nltk.word_tokenize(sentence)


def stem(word): return stemmer.stem(word)


# matrix1 and matrix2 should be numpy array, or broadcasting will not work
def averagePooling(matrix1, matrix2):
  if len(matrix1) > len(matrix2): 
    matrix1, matrix2 = matrix2, matrix1

  windowSize = len(matrix2) - len(matrix1) + 1
  matrixProcessed = np.zeros(len(matrix1))
  for i in range(len(matrix2) - windowSize + 1):
    averaged = np.mean(matrix2[i:i+windowSize], axis=0)
    matrixProcessed = np.vstack((matrixProcessed, averaged))

  return matrix1, matrixProcessed[1:]


def cosineSimilarity(matrix1, matrix2):
  res = 0
  norms1 = LA.norm(matrix1, axis=0)
  norms2 = LA.norm(matrix2, axis=0)
  for i in range(len(matrix1)):
    dotProduct = np.dot(matrix1[i], matrix2[i])
    diff = dotProduct / (norms1[i] * norms2[i])
    res += diff ** 2
  return res