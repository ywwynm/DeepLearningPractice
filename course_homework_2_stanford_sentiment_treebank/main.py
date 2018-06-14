import dataset_imdb as imdb
from string import punctuation
import time, os
from collections import Counter
import numpy as np

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense

def remove_punctuations(sentences):
  ret = []
  for sent in sentences:
    sent_no_punc = ''
    for c in sent:
      if c not in punctuation:
        sent_no_punc += c
    ret.append(sent_no_punc)
  return ret

def get_word_idx_dict(words):
  counter = Counter(words)
  vocab = sorted(counter, key=counter.get, reverse=True)
  return { word: i for i, word in enumerate(vocab, 1) }

def sentence_to_idx(set_no_punc, word_idx_dict):
  return [word_idx_dict[word] for word in set_no_punc.split()]

def sentences_to_idxs(sentences, word_idx_dict):
  return [sentence_to_idx(sent, word_idx_dict) for sent in sentences]

def sentences_idxs_to_net_input(sentences_idxs, max_len):
  ret = np.zeros(shape=(len(sentences_idxs), max_len), dtype=int)
  for i, sent_idx in enumerate(sentences_idxs):
    ret[i, -len(sent_idx):] = np.array(sent_idx)[:max_len]
  return ret

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
  X_train = sentences_idxs_to_net_input(train_sentences_idxs, max_len)
  X_test = sentences_idxs_to_net_input(test_sentences_idxs, max_len)

  Y_train = np.array(train_labels)
  Y_test = np.array(test_labels)

  train_batch_size = 32
  test_batch_size = 32
  train_epochs = 2

  model = Sequential()
  model.add(Embedding(input_dim=num_features, output_dim=256))
  model.add(LSTM(128, dropout=0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  print('start training...')
  start_time = time.time()
  model.fit(X_train, Y_train, batch_size=train_batch_size, epochs=train_epochs, validation_data=(X_test, Y_test))
  print('training finished, costs %.4fs' % (time.time() - start_time))

  print('start testing...')
  start_time = time.time()
  _, acc = model.evaluate(X_test, Y_test, batch_size=test_batch_size)
  print('evaluating finished, costs %.4fs' % (time.time() - start_time))

  print('Test accuracy :', acc)

  model_path = 'IMDB_model'
  if not os.path.exists(model_path):
    os.mkdir(model_path)
  model.save(os.path.join(model_path, 'model.h5'))

  like_review_1 = 'I love this movie'
  dislike_review_1 = 'I do not like this move'
  like_review_2 = 'This movie is attractive and it does well on nearly everything even though there are some minor problems on story'
  dislike_review_2 = 'I have to say that although this movie has some interesting parts it cannot achieve a very good score because it does bad on building impressive characters'
  sentences_to_pred = [like_review_1, dislike_review_1, like_review_2, dislike_review_2]
  sentences_idxs_to_pred = sentences_to_idxs(sentences_to_pred, word_idx_dict)
  X_predict = sentences_idxs_to_net_input(sentences_idxs_to_pred, max_len)
  print(model.predict(X_predict))
