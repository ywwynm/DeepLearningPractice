import scipy.io as scio
from scipy import misc
import tensorflow.python.keras.utils as np_utils

PARENT_PATH = "OXFORD_FLOWERS_17_data"
datasplits = scio.loadmat(PARENT_PATH + "/datasplits.mat")


def get_images_and_labels(datasplit_name):
  indexes = datasplits[datasplit_name]
  images = []
  labels = []
  for i in indexes[0]:
    path = PARENT_PATH + "/17flowers/image_" + str(i).zfill(4) + ".jpg"
    images.append(misc.imread(path))
    if i % 80 == 0:
      label = int(i / 80)
    else:
      label = int(i / 80) + 1
    labels.append(np_utils.to_categorical(label, 17))
  return images, labels