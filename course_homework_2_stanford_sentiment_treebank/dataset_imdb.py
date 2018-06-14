import keras
import os
import time

def load_sentences_from_dir(dir_path):
  sentences = []
  for file_path in os.listdir(dir_path):
    with open(os.path.join(dir_path, file_path), 'r', encoding='utf-8') as f:
      sentences.append(f.readline())
  return sentences

def load_sentences_from_file(file_path):
  with open(file_path, 'r', encoding='utf-8') as f:
    return f.readlines()

def load_labels_from_file(file_path):
  labels = []
  with open(file_path, 'r', encoding='utf-8') as f:
    for line in f.readlines():
      labels.append(int(line))
  return labels

def load_dataset(dir_path):
  pos_dir = os.path.join(dir_path, 'pos')
  pos_sentences = load_sentences_from_dir(pos_dir)
  neg_dir = os.path.join(dir_path, 'neg')
  neg_sentences = load_sentences_from_dir(neg_dir)
  sentences = []
  labels = []
  sentences.extend(pos_sentences)
  labels.extend([1 for _ in range(len(pos_sentences))])
  sentences.extend(neg_sentences)
  labels.extend([0 for _ in range(len(neg_sentences))])
  return sentences, labels

def write_dataset_to_file(file_path, data_arr):
  with open(file_path, 'w', encoding='utf-8') as f:
    for i in range(len(data_arr)):
      data = data_arr[i]
      if i != len(data_arr) - 1:
        f.write(str(data) + '\n')
      else:
        f.write(str(data))

def load_imdb_dataset_maybe_download():
  dataset_dir = 'IMDB_data'
  if os.path.exists(dataset_dir):
    return load_imdb_dataset_from_combined_file()

  print('start getting dataset file by keras...this may cost a long time, please wait patiently, thank you')
  start_time = time.time()
  dataset_path = keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True)
  print('getting dataset file costs %.4fs' % (time.time() - start_time))

  print('loading original dataset files')
  start_time = time.time()
  train_dir_path = os.path.join(os.path.dirname(dataset_path), 'aclImdb', 'train')
  train_sentences, train_labels = load_dataset(train_dir_path)
  test_dir_path = os.path.join(os.path.dirname(dataset_path), 'aclImdb', 'test')
  test_sentences, test_labels = load_dataset(test_dir_path)
  print('load dataset costs %.4fs' % (time.time() - start_time))

  print('writing dataset to combined file')
  os.mkdir(dataset_dir)
  write_dataset_to_file(os.path.join(dataset_dir, 'train_sentences.txt'), train_sentences)
  write_dataset_to_file(os.path.join(dataset_dir, 'train_labels.txt'), train_labels)
  write_dataset_to_file(os.path.join(dataset_dir, 'test_sentences.txt'), test_sentences)
  write_dataset_to_file(os.path.join(dataset_dir, 'test_labels.txt'), test_labels)
  print('write dataset successfully')

  return (train_sentences, train_labels), (test_sentences, test_labels)

def load_imdb_dataset_from_combined_file():
  dataset_dir = 'IMDB_data'
  if not os.path.exists(dataset_dir):
    return load_imdb_dataset_maybe_download()

  train_sentences = load_sentences_from_file(os.path.join(dataset_dir, 'train_sentences.txt'))
  train_labels = load_labels_from_file(os.path.join(dataset_dir, 'train_labels.txt'))
  test_sentences = load_sentences_from_file(os.path.join(dataset_dir, 'test_sentences.txt'))
  test_labels = load_labels_from_file(os.path.join(dataset_dir, 'test_labels.txt'))
  return (train_sentences, train_labels), (test_sentences, test_labels)

# (train_sentences, train_labels), (test_sentences, test_labels) = load_imdb_dataset_maybe_download()