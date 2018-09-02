'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import tensorflow as tf
from keras.utils import multi_gpu_model

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


from keras import backend as K
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import load_model

import os
import pickle
import numpy as np


import gen_conv_net as gcn
# import get_data as gd
import get_vgg16_cifar10 as gvc
import get_wrn as gwrn
import get_lenet as gln

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_load.get_keras_data as gkd




base_folder = "/mnt/additional/nitthilan/data/kaggle/"
df_train_label = base_folder + 'stage_1_train_labels.csv'
df_train_class_info = base_folder + 'stage_1_detailed_class_info.csv'
df_train_base_folder = base_folder + "stage_1_train_images/"
df_test_base_folder = base_folder + "stage_1_test_images/"

df_preprocess_base_folder = base_folder + "preprocess_folder/"
df_train_preprocess_filename = df_preprocess_base_folder + "train_256.npz"
df_test_preprocess_filename = df_preprocess_base_folder + "test_256.npz"


def load_images(filename):
  filevalue = np.load(filename)
  return filevalue["image_array"], filevalue["filename_list"]

def get_inv_mapping(filename_list):
  name_idx_map = {}
  for idx, file in enumerate(filename_list):
    name_idx_map[file] = idx
  return name_idx_map

def get_image_list(filename_list, name_idx_map, image_list):
  image_list = []
  for file in filename_list:
    image_list.append(image_list[name_idx_map[file]])
  return np.array(image_list)

def get_train_info(filename):
  dftl = pd.read_csv(filename)
  dftl_np = dftl.values
  x_list = dftl_np[1]
  y_list = dftl_np[2]
  width_list = dftl_np[3]
  hgt_list = dftl_np[4]
  pred_list = dftl_np[5]
  file_list = dftl_np[0]
  return file_list, x_list, y_list, width_list, hgt_list, pred_list

train_image_list, train_filename_list = \
  load_images(df_train_preprocess_filename)
train_name_idx_map = get_inv_mapping(train_filename_list)
test_image_list, test_filename_list = \
  load_images(df_train_preprocess_filename)
test_name_idx_map = get_inv_mapping(test_filename_list)

train_image_list /= 255
test_image_list /= 255

batch_size = 128 #32
epochs = 200

with tf.device('/gpu:0'):
  # model = gvc.get_conv_net_v1(x_train.shape[1:], \
  #   num_classes, 3-resize_factor) 
  # model = gwrn.create_wide_residual_network(x_train.shape[1:], 
  #   2-resize_factor,
  #   nb_classes=num_classes, N=4, k=8, dropout=0.3)
  model = gcn.get_conv_net(x_train.shape[1:], num_classes) 

print('Not using data augmentation.')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
print('Saved trained model and weights at %s ' % weight_path)



# num_classes = 100 # 10 for cifar10 # 100 for cifar100
# data_augmentation = True
# num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_weight_'


# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

for resize_factor in [0]:#,1,2]:
  # Do not resize input and check the accuracy
  # x_train, y_train, x_test, y_test = \
  #   gd.get_cifar_data(0, num_classes)
  # # x_train, y_train, x_test, y_test = \
  # #   gd.get_cifar10_data(resize_factor)

  # x_train, x_test = gd.scale_image(x_train, x_test)

  x_train, y_train, x_test, y_test = \
    gkd.get_data("mnist") # gkd.get_data("cifar10")
  # x_train, x_test = gd.scale_image(x_train, x_test)
  x_train /= 255
  x_test /= 255
  num_classes = int(y_train.shape[1])


  print(x_train.shape, y_train.shape, \
    x_test.shape, y_test.shape)

  # with tf.device('/cpu:0'):
  with tf.device('/gpu:0'):
    # model = gvc.get_conv_net_v1(x_train.shape[1:], \
    #   num_classes, 3-resize_factor) 
    # model = gwrn.create_wide_residual_network(x_train.shape[1:], 
    #   2-resize_factor,
    #   nb_classes=num_classes, N=4, k=8, dropout=0.3)
    model = gln.get_conv_net(x_train.shape[1:], num_classes) 

    # model = gcn.get_conv_net(x_train.shape[1:], \
    #   num_classes, 2-resize_factor)

  # parallel_model = multi_gpu_model(model, gpus=4)
  parallel_model = model

  print('Not using data augmentation.')
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True)
  print('Saved trained model and weights at %s ' % weight_path)

