import os, time
import matplotlib.pyplot as plt


def save_result(epochs_arr, losses, epochs_10_arr, accuracies):
  if not os.path.exists("result"):
    os.mkdir("result")

  post_fix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

  plt.plot(epochs_arr, losses)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.savefig("result/alex_net_loss_" + post_fix + ".png")

  plt.clf()  # clear existing figure content
  plt.plot(epochs_10_arr, accuracies)
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.savefig("result/alex_net_acc_" + post_fix + ".png")