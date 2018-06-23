import os

import helper
from dataset import classes_names

from PIL import Image
import numpy as np
import pandas as pd

test_data_dir = 'test'

if __name__ == '__main__':
  model_path = helper.train_and_evaluate()
  model = helper.get_model(model_path)

  img_names = []
  img_pils = []
  for img_name in os.listdir(test_data_dir):
    img = Image.open(os.path.join(test_data_dir, img_name))
    img = img.convert('RGB')
    img_names.append(img_name)
    img_pils.append(img)

  predicts = helper.predict(model, img_pils)  # outputs are classes indexes
  predict_classes = [classes_names[cls] for cls in predicts]  # names here

  # img_names = ['1.png', '2.png']
  # predict_classes = ['hello', 'world']

  pd_data = {'file': img_names, 'species': predict_classes}
  df = pd.DataFrame(pd_data)
  df.to_csv('output.csv', index=False)