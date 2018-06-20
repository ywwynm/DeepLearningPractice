import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import utils

classes_names = [
  'Black-grass',
  'Charlock',
  'Cleavers',
  'Common Chickweed',
  'Common wheat',
  'Fat Hen',
  'Loose Silky-bent',
  'Maize',
  'Scentless Mayweed',
  'Shepherds Purse',
  'Small-flowered Cranesbill',
  'Sugar beet'
]

train_data_dir = 'train'

class PlantDataset(Dataset):

  def __init__(self, transform=None):
    self.transform = transform

    self.names = []
    self.X = []
    self.Y = []

    for dir_name in os.listdir(train_data_dir):
      label = classes_names.index(dir_name)  # todo add 1 here?
      for img_name in os.listdir(os.path.join(train_data_dir, dir_name)):
        img = Image.open(os.path.join(train_data_dir, dir_name, img_name))
        img = img.convert('RGB')
        self.names.append(img_name)
        self.X.append(img)
        self.Y.append(label)

  def __getitem__(self, idx):
    x = self.X[idx]
    if self.transform is not None:
      x = self.transform(x)
    return self.names[idx], x, self.Y[idx]

  def __len__(self):
    return len(self.X)


def get_train_validation_data_loader(
    resize_size, batch_size, random_seed, augment=False,
    validation_size=0.3, shuffle=True, show_sample=False):

  # normalize = transforms.Normalize(
  #   mean=[0.485, 0.456, 0.406],
  #   std=[0.229, 0.224, 0.225],
  # )

  # images are all square
  if augment:
    # just use one kind of augmentation for each image
    transforms_random_apply = transforms.RandomApply([
      transforms.RandomChoice([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomRotation(60)
      ]),
    ], p=0.4)
    train_transform = transforms.Compose([
      transforms.Resize(resize_size),
      transforms_random_apply,
      transforms.ToTensor(),
      # normalize
    ])
  else:
    train_transform = transforms.Compose([
      transforms.Resize(resize_size),
      transforms.ToTensor(),
      # normalize
    ])

  valid_transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.ToTensor(),
    # normalize
  ])

  train_dataset = PlantDataset(transform=train_transform)
  valid_dataset = PlantDataset(transform=valid_transform)

  num_train = len(train_dataset)
  indices = list(range(num_train))
  split = int(np.floor(validation_size * num_train))

  if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

  train_idx, valid_idx = indices[split:], indices[:split]
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=4)

  # visualize some images
  if show_sample:
    sample_loader = DataLoader(train_dataset, batch_size=9, shuffle=shuffle)
    data_iter = iter(sample_loader)
    names, Xs, Ys = data_iter.next()
    Xs = Xs.numpy().transpose([0, 2, 3, 1])
    utils.plot_images(classes_names, Xs, Ys)

  return train_loader, valid_loader

