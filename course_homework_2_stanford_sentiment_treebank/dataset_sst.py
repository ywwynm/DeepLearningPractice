
parent_path = 'STANFORD_SENTIMENT_TREEBANK_data/'
sentences_path = parent_path + 'datasetSentences.txt'
dictionary_path = parent_path + 'dictionary.txt'
labels_path = parent_path + 'sentiment_labels.txt'
split_path = parent_path + 'datasetSplit.txt'

def get_sentences():
  ret = []
  with open(sentences_path, 'r', encoding='utf-8') as f:
    _ = next(f)
    for line in f:
      arr = line.split('\t', 1)
      ret.append(arr[1].replace('\n', ''))
  return ret

def get_dictionary():
  ret = {}
  with open(dictionary_path, 'r', encoding='latin-1') as f:
    for line in f:
      arr = line.replace('\n', '').split('|')
      ret[arr[0]] = int(arr[1])
  return ret

def get_scores():
  ret = {}
  with open(labels_path, 'r') as f:
    _ = next(f)
    for line in f:
      arr = line.split('|')
      ret[int(arr[0])] = float(arr[1])
  return ret

def get_sentences_sentiment():
  """
  Get sentiment, instead of score, of sentences
  1, 2, 3, 4 and 5 for very negative, negative, neutral, positive and very positive
  :return: sentiment of sentences
  """
  dictionary = get_dictionary()
  sentences = get_sentences()
  scores = get_scores()
  ret = {}
  for sentence in sentences:
    idx = dictionary[sentence]
    score = scores[idx]
    if score <= 0.2:
      label = 1
    elif score <= 0.4:
      label = 2
    elif score <= 0.6:
      label = 3
    elif score <= 0.8:
      label = 4
    else:
      label = 5
    ret[sentence] = label
  return ret


def get_data_split_dict():
  ret = {}
  with open(split_path, 'r') as f:
    _ = next(f)  # ignore first line
    for line in f:
      arr = line.split(',')
      ret[int(arr[0])] = int(arr[1])
  return ret


print(get_sentences_sentiment())