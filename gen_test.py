from gensim import models
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.snowball import SnowballStemmer
####### url: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/ #########
stemmer = SnowballStemmer("english")

# nltk.download('punkt')
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec', 'computing'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
# model = models.Word2Vec(sentences, min_count=1, size=10, window=2)
# # summarize the loaded model
# words = list(model.wv.vocab)
# print(words)
# print(model['second'])
tokenized = word_tokenize(' '.join(sentences[0]))
print(list(map(lambda x : stemmer.stem(x.lower()), tokenized)))



