import nltk
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from numpy import linalg as LA

stemmer = SnowballStemmer('english')

def tokenize(sentence): return nltk.word_tokenize(sentence)


def stem(word): return stemmer.stem(word)


# matrix1 and matrix2 should be numpy array, or broadcasting will not work
def averagePooling(matrix1, matrix2):
  # let matrix1 be the shorter array, matrix2 be the longer
  if len(matrix1) > len(matrix2): 
    matrix1, matrix2 = matrix2, matrix1

  windowSize = len(matrix2) - len(matrix1) + 1  # get the window size for sliding
  matrixProcessed = np.zeros(len(matrix1))      # some useless initialization, that's why the return processed matrix is sliced from [1:]

  # sliding process begins
  for i in range(len(matrix2) - windowSize + 1):
    averaged = np.mean(matrix2[i:i+windowSize], axis=0)
    matrixProcessed = np.vstack((matrixProcessed, averaged))

  return matrix1, matrixProcessed[1:]


def cosineSimilarity(matrix1, matrix2):
  # matrix one and two should be of same length, if not, pass matrices into averagePooling
  if len(matrix1) != len(matrix2):
    matrix1, matrix2 = averagePooling(matrix1, matrix2)

  res = 0       # l2 norm for these two matrices

  # norm of each 1d arr in the 2d matrix
  norms1 = LA.norm(matrix1, axis=1)
  norms2 = LA.norm(matrix2, axis=1)

  # iterating each 1d word feature arr to compute cos_sim, and add to l2 norm
  for i in range(len(matrix1)):
    dotProduct = np.dot(matrix1[i], matrix2[i])
    diff = dotProduct / (norms1[i] * norms2[i])
    res += diff ** 2
  return res
