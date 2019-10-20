#for converting time series stock data into recurrent graphs
import numpy as np 
import pandas as pd 
from scipy.spatial.distance import pdist, squareform 
import sklearn as sk
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
import os
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import time
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils

#this function is responsible for generating the recurrent graphs
def plot_a_recurrence(s, eps=None, steps=None):
    if eps==None: 
      eps=0.1
    if steps==None: 
      steps=10
    d = sk.metrics.pairwise.pairwise_distances(s)
    d = np.floor(d / eps)
    d[d > steps] = steps
    return d
 
#choosing the value of stock at the ending of the stock day, i.e, chosing the column Close in the dataset
column_names = ["Close"]
fig = plt.figure(figsize=(15,14))
ax = fig.add_subplot(1, 1, 1)
# this line computes the recurrent graph for the entire data set. We can change the data set on need basis to get the corresponding grpahs
ax.imshow(plot_a_recurrence(pd.read_csv("train_data.csv")[column_names].values,steps=1000))
