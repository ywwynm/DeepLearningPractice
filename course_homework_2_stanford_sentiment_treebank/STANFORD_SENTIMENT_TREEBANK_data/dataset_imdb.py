import tensorflow as tf
import os

def load_sentences_from_dir(dir_path):
  sentences = []
  for file_path in os.listdir(dir_path):
    with open(file_path, 'r') as f:
      sentences.append(f.readline())
  return sentences

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

def load_imdb_dataset_maybe_download():
  dataset_path = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True)

  train_dir_path = os.path.join(os.path.dirname(dataset_path), 'aclImdb', 'train')
  train_sentences, train_labels = load_dataset(train_dir_path)
  test_dir_path = os.path.join(os.path.dirname(dataset_path), 'aclImdb', 'test')
  test_sentences, test_labels = load_dataset(test_dir_path)
  return (train_sentences, train_labels), (test_sentences, test_labels)

(train_sentences, train_labels), (test_sentences, test_labels) = load_imdb_dataset_maybe_download()
print(1)