import time, os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import *

import dataset

random_seed = 96
validation_size = 0.3
eval_epoch_step = 1

num_epoch = 32
batch_size = 32
lr = 1e-3
weight_decay = 5e-4


def __replace_model_fc(model):
  model.fc = nn.Linear(512 * 1, 200)


def __get_model_params(model, only_fc=False):
  if only_fc:
    return model.fc.parameters()
  else:
    return model.parameters()


def get_model(saved_model_path=None):
  model = resnet18(pretrained=True)
  __replace_model_fc(model)
  if saved_model_path is not None:
    model.load_state_dict(torch.load(saved_model_path))
  model = model.cuda()
  return model


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
      predict = model(Xs)
      _, predicted = torch.max(predict.data, 1)
      total_num += Xs.size(0)
      correct_num += (predicted == Ys).sum().item()

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
  for epoch in range(num_epoch):
    running_loss = 0.0
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

      print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, loss.item()))
      print('cost time: %.4fs' % (time.time() - start_time))

    if epoch == 0 or (epoch + 1) % eval_epoch_step == 0:
      print('evaluating on train set during training, epoch: %d' % epoch)
      __evaluate(model, train_loader)
      print('evaluating on validation set during training, epoch: %d' % epoch)
      val_acc = __evaluate(model, valid_loader)
      if val_acc >= max_val_acc:
        max_val_acc = val_acc
        best_state_dict = model.state_dict()

  print('model is trained')
  print('evaluating on validation set')
  __evaluate(model, valid_loader)

  print('saving model...')
  if not os.path.exists('models'):
    os.mkdir('models')
  model_path = os.path.join('models', 'model-acc%.2f-%s.pth' % (max_val_acc, time.strftime('M-d-H:mm')))
  torch.save(best_state_dict, model_path)
  return model_path


def train_and_evaluate():
  start_time = time.time()
  print('loading dataset...')
  train_loader, valid_loader = dataset.get_train_validation_data_loader(
    resize_size=(224, 224), batch_size=batch_size, random_seed=random_seed, show_sample=False)
  print('dataset loaded, cost time: %.4fs' % (time.time() - start_time))

  model = get_model()

  print('fine tuning fc layer..')
  start_time = time.time()
  __train_and_evaluate(model, [train_loader, valid_loader], only_fc=True)
  print('fc layer tuned, cost time: %.4fs' % (time.time() - start_time))

  print('\nfine tuning all layers')
  start_time = time.time()
  __train_and_evaluate(model, [train_loader, valid_loader], only_fc=False)
  print('all layers tuned, cost time: %.4fs' % (time.time() - start_time))

