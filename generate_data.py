import numpy as np
import random
import torch
from util.transformer import Transformer
import pickle

import matplotlib.pyplot as plt

transformer = Transformer()

clean_data = None
clean_label = None

mean = np.array((0.4914, 0.4822, 0.4465)).reshape(1,3,1)
var = np.array((0.2023, 0.1994, 0.2010)).reshape(1,3,1)

for i in range(1,6):
    with open('data/cifar-10/data_batch_' + str(i), 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
        if clean_data is None:
            temp = dict['data'].reshape(10000,3,32,32).astype('float32')
            #print('temp is ', temp)
            for j in range(temp.shape[0]):
                img = temp[j]
                img = transformer.random_crop(img, (32, 32), padding=4)
                img = transformer.random_horizontal_flip(img)
                temp[j] = img
            temp = temp.reshape(10000,3,-1)
            clean_data = ((temp / 255 - mean) / var).reshape(10000,-1)
            #print('clean is ', clean_data)
            clean_label = np.array(dict['labels']).astype('int64').reshape(10000, 1)
        else:
            temp = dict['data'].reshape(10000, 3, 32, 32).astype('float32')
            for j in range(temp.shape[0]):
                img = temp[j]
                img = transformer.random_crop(img, (32, 32), padding=4)
                img = transformer.random_horizontal_flip(img)
                temp[j] = img
            temp = temp.reshape(10000, 3, -1)
            temp = ((temp / 255 - mean) / var).reshape(10000,-1)
            clean_data = np.vstack((clean_data, temp))
            clean_label = np.vstack((clean_label, np.array(dict['labels']).astype('int64').reshape(10000, 1)))

clean_data = clean_data.reshape(-1,3,32,32)
np.savez('data/train' ,data=clean_data, label=np.squeeze(clean_label))

with open('data/cifar-10/test_batch', 'rb') as fo:
    dict = pickle.load(fo, encoding='latin1')
    temp = dict['data'].reshape(10000, 3, -1).astype('float32')
    test_data = ((temp / 255 - mean) / var).reshape(10000,-1)
    test_label = np.array(dict['labels']).astype('int64')

test_data = test_data.reshape(-1,3,32,32)

size = test_data.shape[0]
sample_size = 1000

idx = np.random.choice(range(size), sample_size, replace=False).tolist()

valid_data, valid_label = test_data[idx], test_label[idx]
test_data, test_label = np.delete(test_data, idx, axis=0), np.delete(test_label, idx, axis=0)

np.savez('data/valid' ,data=valid_data, label=valid_label)
np.savez('data/test', data=test_data, label=test_label)