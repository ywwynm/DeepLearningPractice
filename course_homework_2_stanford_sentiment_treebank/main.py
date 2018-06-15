import dataset_imdb as imdb
from string import punctuation
import time, os
from collections import Counter
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.callbacks import TensorBoard

EMB_DIM = 200
glove_txt_path = 'glove.6B.200d.txt'

def get_embedding_matrix(word_idx_dict):
  embedding_vectors = {}
  with open(glove_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
      values = line.split()
      word = values[0]
      floats = np.asarray(values[1:], dtype=np.float32)
      embedding_vectors[word] = floats
  matrix = np.zeros(shape=(len(word_idx_dict) + 1, EMB_DIM))
  for word in word_idx_dict.keys():
    idx = word_idx_dict[word]
    try:
      emb_vec = embedding_vectors[word]
      matrix[idx] = emb_vec
    except KeyError:
      # just ignored
      print('KeyError: ' + word)
      continue
  return matrix


# 对文本的处理

def lower_and_remove_punctuations(sentences):  # 去除句子中的标点符号并全部转换成小写
  ret = []
  for sent in sentences:
    sent_no_punc = ''
    for c in sent:
      if c not in punctuation:
        sent_no_punc += c
    ret.append(sent_no_punc.lower())
  return ret

def get_word_idx_dict(words):  # 获得单词-序号词典
  counter = Counter(words)
  print(counter)
  vocab = sorted(counter, key=counter.get, reverse=True)
  return { word: i for i, word in enumerate(vocab, 1) }

def sentence_to_idxs(set_no_punc, word_idx_dict):  # 将一个句子转换为序号序列
  return [word_idx_dict[word] for word in set_no_punc.split()]

def sentences_to_idxs_list(sentences, word_idx_dict):  # 将多个句子转换为序号序列
  return [sentence_to_idxs(sent, word_idx_dict) for sent in sentences]

def sentences_idxs_list_to_net_input(sentences_idxs_list, max_len):
  # 将多个句子序号序列统一长度，转换为网络的输入
  ret = np.zeros(shape=(len(sentences_idxs_list), max_len), dtype=int)
  for i, sent_idx in enumerate(sentences_idxs_list):
    ret[i, -len(sent_idx):] = np.array(sent_idx)[:max_len]
  return ret

class BatchTensorBoard(TensorBoard):

  def __init__(self, log_batch_step=1, **kwargs):
    super().__init__(**kwargs)
    self.log_batch_step = log_batch_step
    self.batch_count = 0

  def on_batch_end(self, batch, logs=None):
    self.batch_count += 1
    if self.batch_count % self.log_batch_step == 0:
      for name, value in logs.items():
        if name in ['batch', 'size']:
          continue
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value.item()
        summary_value.tag = name
        self.writer.add_summary(summary, self.batch_count)
      self.writer.flush()

    super().on_batch_end(batch, logs)

if __name__ == '__main__':
  (train_sentences, train_labels), (test_sentences, test_labels) = imdb.load_imdb_dataset_maybe_download()

  print('lowering and removing punctuations for sentences...')
  start_time = time.time()
  train_sentences = lower_and_remove_punctuations(train_sentences)
  test_sentences = lower_and_remove_punctuations(test_sentences)
  print('sentences were lowered and punctuations were removed, costs %.4fs' % (time.time() - start_time))

  all_sentences = []
  all_sentences.extend(train_sentences)
  all_sentences.extend(test_sentences)
  sentences_lens_counter = Counter([len(sent) for sent in all_sentences])
  print('max sentence length: %d' % max(sentences_lens_counter))

  print('generating all text...')
  start_time = time.time()
  all_text = ''.join(sent + ' ' for sent in all_sentences)
  print('all text generated, costs %.4fs' % (time.time() - start_time))

  words = all_text.split()
  word_idx_dict = get_word_idx_dict(words)
  num_features = len(word_idx_dict)
  print('features(total words) num: %d' % num_features)

  print('indexing sentences...')
  start_time = time.time()
  train_sentences_idxs = sentences_to_idxs_list(train_sentences, word_idx_dict)
  test_sentences_idxs = sentences_to_idxs_list(test_sentences, word_idx_dict)
  print('sentences indexed, costs %.4fs' % (time.time() - start_time))

  max_len = 360
  X_train = sentences_idxs_list_to_net_input(train_sentences_idxs, max_len)
  X_test = sentences_idxs_list_to_net_input(test_sentences_idxs, max_len)

  Y_train = np.array(train_labels)
  Y_test = np.array(test_labels)

  train_batch_size = 32
  test_batch_size = 32
  train_epochs = 2

  use_glove = True
  model = Sequential()
  if use_glove:
    model.add(Embedding(input_dim=num_features + 1, output_dim=EMB_DIM,
                        weights=[get_embedding_matrix(word_idx_dict)], input_length=max_len, trainable=False))
  else:
    model.add(Embedding(input_dim=num_features + 1, output_dim=EMB_DIM, input_length=max_len))
  model.add(LSTM(128, dropout=0.5))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  tensorboard_callback = BatchTensorBoard(batch_size=train_batch_size, write_images=True)
  print('start training...')
  start_time = time.time()
  model.fit(X_train, Y_train, batch_size=train_batch_size, epochs=train_epochs,
            validation_data=(X_test, Y_test), callbacks=[tensorboard_callback])
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

  like_review_1 = 'I love this movie'.lower()
  dislike_review_1 = 'I do not like this move'.lower()
  like_review_2 = 'This movie is attractive and it does well on nearly everything ' \
                  'even though there are some minor problems on story'.lower()
  dislike_review_2 = 'I have to say that although this movie has some interesting parts ' \
                     'it cannot achieve a very good score ' \
                     'because it does bad on building impressive characters'.lower()
  sentences_to_pred = [like_review_1, dislike_review_1, like_review_2, dislike_review_2]
  sentences_idxs_to_pred = sentences_to_idxs_list(sentences_to_pred, word_idx_dict)
  X_predict = sentences_idxs_list_to_net_input(sentences_idxs_to_pred, max_len)
  print(model.predict(X_predict))
