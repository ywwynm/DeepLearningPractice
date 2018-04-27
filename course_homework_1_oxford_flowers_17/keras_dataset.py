import scipy.io as scio
from scipy import misc
import keras.utils as np_utils
import numpy as np

PARENT_PATH = "OXFORD_FLOWERS_17_data"
datasplits = scio.loadmat(PARENT_PATH + "/datasplits.mat")


def get_images_and_labels(datasplit_name):
  indexes = datasplits[datasplit_name]
  images = []
  labels = []
  for i in indexes[0]:
    path = PARENT_PATH + "/17flowers/image_" + str(i).zfill(4) + ".jpg"
    images.append(misc.imresize(misc.imread(path), [224, 224]))
    if i % 80 == 0:
      label = int(i / 80) - 1
    else:
      label = int(i / 80)
    labels.append(label)
  return np.asarray(images), np_utils.to_categorical(labels, 17)