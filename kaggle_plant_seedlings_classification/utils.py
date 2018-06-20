import matplotlib.pyplot as plt

def plot_images(class_names, images, labels_true, labels_pred=None):
  """
  Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
  images should be a numpy array
  """
  fig, axes = plt.subplots(3, 3)
  for i, ax in enumerate(axes.flat):
    # plot img
    ax.imshow(images[i, :, :, :], interpolation='spline16')
    # show true & predicted classes
    cls_true_name = class_names[labels_true[i]]
    if labels_pred is None:
      xlabel = "{0} ({1})".format(cls_true_name, labels_true[i])
    else:
      cls_pred_name = class_names[labels_pred[i]]
      xlabel = "True: {0}\nPred: {1}".format(
        cls_true_name, cls_pred_name
      )
    ax.set_xlabel(xlabel)
    ax.set_xticks([])
    ax.set_yticks([])
  plt.show()