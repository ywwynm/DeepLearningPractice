import scipy.io as scio
import tensorflow as tf

PARENT_PATH = "OXFORD_FLOWERS_17_data"
datasplits = scio.loadmat(PARENT_PATH + "/datasplits.mat")

def construct_dataset(datasplits_name):
  indexes = datasplits[datasplits_name]

  def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [224, 224])
    return image_resized, label

  img_paths = []
  labels = []
  for i in indexes[0]:
    img_paths.append(PARENT_PATH + "/17flowers/image_" + str(i).zfill(4) + ".jpg")
    if i % 80 == 0:
      labels.append(int(i / 80))
    else:
      labels.append(int(i / 80) + 1)

  dataset = tf.data.Dataset.from_tensor_slices((tf.constant(img_paths), tf.constant(labels)))
  return dataset.map(_parse_function)

def get_train_set_1():
  return construct_dataset('trn1')

def get_validation_set_1():
  return construct_dataset('val1')
