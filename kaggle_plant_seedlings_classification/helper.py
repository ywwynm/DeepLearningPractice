import time, os, copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import dataset
from bcnn import BilinearResNet34

random_seed = 96  # for splitting train set and validation set
validation_size = 0.3
eval_epoch_step = 4

use_bilinear = True
use_normalize = False
resize_size = 448
augment = True

num_epoch = 10  # will be changed when fine tuning fc and all layers
batch_size = 32
lr = 1e-3
weight_decay = 5e-4

save_model = False  # will be changed when fine tuning fc and all layers

outputs_dir = os.path.join('outputs', 'output_' + time.strftime("%m-%d-%H-%M", time.localtime()))


def __replace_model_fc(model):
  model.fc = nn.Linear(512 * 1, 12)


def __get_model_params(model, only_fc=False):
  if only_fc:
    return model.fc.parameters()
  else:
    return model.parameters()


def get_model(saved_model_path=None):
  if use_bilinear:
    model = BilinearResNet34()
  else:
    model = resnet34(pretrained=True)
    __replace_model_fc(model)

  if saved_model_path is not None:
    model.load_state_dict(torch.load(saved_model_path))
  model = model.cuda()
  return model


def __save_plot(Y_values, name, multiply_epoch_step=True):
  if multiply_epoch_step:
    X_values = [(i + 1) * eval_epoch_step for i in range(len(Y_values))]
  else:
    X_values = [i + 1 for i in range(len(Y_values))]

  plt.clf()
  plt.plot(X_values, Y_values)
  plt.xlabel('epoch')
  plt.ylabel(name)
  plt.savefig(os.path.join(outputs_dir, name + '_' + time.strftime("%m-%d-%H-%M", time.localtime()) + '.png'))


def __evaluate(model, data_loader):
  model.eval()  # evaluation mode

  start_time = time.time()
  correct_num = 0
  total_num = 0
  with torch.no_grad():
    for data in data_loader:
      _, Xs, Ys = data
      Xs = Xs.cuda()
      Ys = Ys.cuda()
      output = model(Xs)
      predicts = torch.argmax(output.data, 1)
      total_num += Xs.size(0)
      correct_num += (predicts == Ys).sum().item()

  model.train()  # back to train mode

  acc = 100.0 * correct_num / total_num
  print('accuracy: %.4f%%, cost time: %.4fs' % (acc, time.time() - start_time))
  return acc


def __train_and_evaluate(model, loaders, only_fc=False):
  train_loader = loaders[0]
  valid_loader = loaders[1]
  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = optim.SGD(
    __get_model_params(model, only_fc),
    lr=lr,
    momentum=0.9,
    weight_decay=weight_decay
  )

  print('start training...')

  max_val_acc = 0.0
  best_state_dict = None

  trn_losses_arr = []
  trn_acc_arr = []
  val_acc_arr = []

  for epoch in range(num_epoch):
    running_loss = 0.0
    batch_num = 0
    for i, (_, Xs, Ys) in enumerate(train_loader, 0):
      start_time = time.time()

      Xs = Xs.cuda()
      Ys = Ys.cuda()

      optimizer.zero_grad()
      outputs = model(Xs)
      loss = criterion(outputs, Ys)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      batch_num = i

      print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, loss.item()))
      print('cost time: %.4fs' % (time.time() - start_time))

    trn_losses_arr.append(running_loss / batch_num)

    if (epoch + 1) % eval_epoch_step == 0:
      print('evaluating on train set during training, epoch: %d' % (epoch + 1))
      trn_acc = __evaluate(model, train_loader)
      trn_acc_arr.append(trn_acc)

      print('evaluating on validation set during training, epoch: %d' % (epoch + 1))
      val_acc = __evaluate(model, valid_loader)
      val_acc_arr.append(val_acc)
      if val_acc >= max_val_acc:
        max_val_acc = val_acc
        best_state_dict = copy.deepcopy(model.state_dict())

  print('model is trained')

  model.load_state_dict(best_state_dict)
  print('evaluating on validation set with best model')
  __evaluate(model, valid_loader)

  __save_plot(trn_losses_arr, 'trn_loss', multiply_epoch_step=False)
  __save_plot(trn_acc_arr, 'trn_acc')
  __save_plot(val_acc_arr, 'val_acc')

  if save_model:
    print('saving model...')
    model_path = os.path.join(outputs_dir, 'model.pth')
    torch.save(best_state_dict, model_path)
    return model_path
  else: return ""


def predict(model, img_pils):
  model.eval()
  predicts = []

  if use_normalize:
    normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225],
    )
  else:
    normalize = transforms.Normalize(
      mean=[0.0, 0.0, 0.0],
      std=[1.0, 1.0, 1.0],
    )
  transform = transforms.Compose([
    transforms.Resize(size=(resize_size, resize_size)),
    transforms.ToTensor(),
    normalize
  ])

  with torch.no_grad():
    for img_pil in img_pils:
      img_tensor = transform(img_pil)
      img_tensor = img_tensor.unsqueeze(0).cuda()
      output = model(img_tensor)
      predict = torch.argmax(output.data, 1)
      predicts.append(predict.item())
  return predicts


def train_and_evaluate():
  if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

  start_time = time.time()
  print('loading dataset...')
  train_loader, valid_loader = dataset.get_train_validation_data_loader(
    resize_size=(resize_size, resize_size), batch_size=batch_size, random_seed=random_seed,
    use_normalize=use_normalize, augment=augment, show_sample=False)
  print('dataset loaded, cost time: %.4fs' % (time.time() - start_time))

  model = get_model()

  global save_model, num_epoch

  print('fine tuning fc layer..')
  num_epoch = 100
  save_model = False
  start_time = time.time()
  __train_and_evaluate(model, [train_loader, valid_loader], only_fc=True)
  print('fc layer tuned, cost time: %.4fs' % (time.time() - start_time))

  print('\nfine tuning all layers')
  num_epoch = 200
  save_model = True
  start_time = time.time()
  model_path = __train_and_evaluate(model, [train_loader, valid_loader], only_fc=False)
  print('all layers tuned, cost time: %.4fs' % (time.time() - start_time))
  return model_path

