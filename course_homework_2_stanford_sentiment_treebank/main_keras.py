# import torchtext.data as data
# import torchtext.datasets as datasets
# from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
#
# TEXT = data.Field()
# LABEL = data.Field(sequential=True)
# train, val, test = datasets.SST.splits(
#     TEXT, LABEL, fine_grained=True, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')
# print('train.fields', train.fields)
# print('len(train)', len(train))
# print('vars(train[0])', vars(train[0]))
#
# # build the vocabulary
# url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
# TEXT.build_vocab(train, vectors=Vectors('wiki.simple.vec', url=url))
# LABEL.build_vocab(train)
#
# # print vocab information
# print('len(TEXT.vocab)', len(TEXT.vocab))
# print('TEXT.vocab.vectors.size()', TEXT.vocab.vectors.size())

from keras.utils import get_file
import tensorflow as tf
import keras.datasets.imdb as imdb
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 96
np.random.seed(seed)

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

