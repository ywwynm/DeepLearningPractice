import dataset_imdb as imdb
from string import punctuation
import time
from collections import Counter
import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense

def remove_punctuations(sentences):
  ret = []
  for set in sentences:
    set_no_punc = ''
    for c in set:
      if c not in punctuation:
        set_no_punc += c
    ret.append(set_no_punc)
  return ret

def get_word_idx_dict(words):
  counter = Counter(words)
  vocab = sorted(counter, key=counter.get, reverse=True)
  return { word: i for i, word in enumerate(vocab, 1) }

def sentence_to_idx(set_no_punc, word_idx_dict):
  return [word_idx_dict[word] for word in set_no_punc.split()]

def sentences_to_idxs(sentences, word_idx_dict):
  return [sentence_to_idx(set, word_idx_dict) for set in sentences]

if __name__ == '__main__':
  (train_sentences, train_labels), (test_sentences, test_labels) = imdb.load_imdb_dataset_maybe_download()

  print('removing punctuations for sentences...')
  start_time = time.time()
  train_sentences = remove_punctuations(train_sentences)
  test_sentences = remove_punctuations(test_sentences)
  print('punctuations removed, costs %.4fs' % (time.time() - start_time))

  all_sentences = []
  all_sentences.extend(train_sentences)
  all_sentences.extend(test_sentences)

  print('generating all text...')
  start_time = time.time()
  all_text = ''.join(set + ' ' for set in all_sentences)
  print('all text generated, costs %.4fs' % (time.time() - start_time))

  words = all_text.split()
  word_idx_dict = get_word_idx_dict(words)
  num_features = len(word_idx_dict)

  print('indexing sentences...')
  start_time = time.time()
  train_sentences_idxs = sentences_to_idxs(train_sentences, word_idx_dict)
  test_sentences_idxs = sentences_to_idxs(test_sentences, word_idx_dict)
  print('sentences indexed, costs %.4fs' % (time.time() - start_time))

  max_len = 360
  X_train = np.zeros(shape=(len(train_sentences_idxs), max_len), dtype=int)
  for i, set in enumerate(train_sentences_idxs):
    X_train[i, -len(set):] = np.array(set)[:max_len]
  X_test = np.zeros(shape=(len(test_sentences_idxs), max_len), dtype=int)
  for i, set in enumerate(test_sentences_idxs):
    X_test[i, -len(set):] = np.array(set)[:max_len]

  Y_train = np.array(train_labels)
  Y_test = np.array(test_labels)

  train_batch_size = 16
  test_batch_size = 16
  train_epochs = 2

  model = Sequential()
  model.add(Embedding(input_dim=num_features, output_dim=256))
  model.add(LSTM(128, dropout=0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.fit(X_train, Y_train, batch_size=train_batch_size, epochs=train_epochs, validation_data=(X_test, Y_test))
  score, acc = model.evaluate(X_test, Y_test, batch_size=test_batch_size)
  print('Test score: ', score)
  print('Test accuracy :', acc)