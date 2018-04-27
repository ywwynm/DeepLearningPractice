import tensorflow as tf
import scipy.io as scio

PARENT_PATH = "OXFORD_FLOWERS_17_data"
datasplits = scio.loadmat(PARENT_PATH + "/datasplits.mat")

def construct_dataset(datasplits_name):
  indexes = datasplits[datasplits_name]

  def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 224, 224)  # new shape is [224, 224, 3]
    image_float = tf.image.per_image_standardization(image_resized)
    label_one_hot = tf.one_hot(label, 17, 1, 0)
    # return tf.cast(image_resized, tf.float32) / 255.0, label_one_hot
    return image_float, label_one_hot

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

def get_train_set(idx):
  return construct_dataset('trn' + str(idx))

def get_validation_set(idx):
  return construct_dataset('val' + str(idx))

def get_test_set(idx):
  return construct_dataset('tst' + str(idx))
