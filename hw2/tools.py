import os
import random 
import time
import numpy as np
import pandas as pd

def rescaling(data):
    '''
    Arrange all feature btw [0, 1]
    '''
    if data.ndim > 2:
        raise('不支援 2 維以上的 array 喔喔喔喔喔喔喔')
    else:
        rmax, rmin = np.amax(data, axis = 0), np.amin(data, axis = 0)
        new_data = (data - rmin) / (rmax - rmin)
    return new_data, rmax, rmin

def scaling(data, smax, smin):
    '''
    Arrange all feature btw [0, 1]
    '''
    if data.ndim > 2:
        raise('不支援 2 維以上的 array 喔喔喔喔喔喔喔')
    else:
        new_data = (data - smin) / (smax - smin)
    return new_data

def shuffle(x, y):
  '''
  # shuffle x 和 y
  '''
  fea_num = x.shape[1]
  data = np.insert(x, fea_num, y, axis =1)
  np.random.shuffle(data)
  x = data[:, :fea_num]
  y = data[:, fea_num:]
  return x, y.flatten()

def savemodel():
  return 0

def getfeature(featurelist,x):
  if featurelist[0] == -1:
    return x
  tempX = np.zeros((len(x[:,0]),1))
  for f in featurelist:
    tempX = np.concatenate((tempX,np.expand_dims(x[:,f], axis=1)),axis = 1)
  tempX = tempX[:,1:]
  return tempX

def addxn(xnlist,x,exp):
  if xnlist[0] == -1:
    for col in range(x.shape[1]):
      x = np.concatenate((x,np.expand_dims(x[:,col], axis=1)**exp),axis = 1)
  else:
    for col in xnlist:
      x = np.concatenate((x,np.expand_dims(x[:,col], axis=1)**exp),axis = 1)
  return x

def p(m):
  print(m)
  exit()