import keras
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import csv
import pickle
def normalize_rmse(y, y_pred):
  y = y * 1.116897661 + 3.581712086
  # y_pred = keras.backend.clip(y_pred, 1.0, 5.0)
  y_pred = y_pred * 1.116897661 + 3.581712086  
  return keras.backend.sqrt(keras.backend.mean((y_pred-y) ** 2))

def rmse(y, y_pred):
  # y = y * 1.116897661 + 3.581712086
  y_pred = keras.backend.clip(y_pred, 1.0, 5.0)
  # y_pred = y_pred * 1.116897661 + 3.581712086  
  return keras.backend.sqrt(keras.backend.mean((y_pred-y) ** 2))

def p_normalize_rmse(y, y_pred):
  y = y[0] * y[2] + y[1]
  y_pred = y_pred[0] * y_pred[2] + y_pred[1]
  # y_pred = keras.backend.clip(y_pred, 1.0, 5.0)
  return keras.backend.sqrt(keras.backend.mean((y_pred-y) ** 2))

def non_normalize_rmse(y, y_pred):
  return keras.backend.sqrt(keras.backend.mean((y_pred-y) ** 2))

class Sig5(Activation):
    def __init__(self, activation, **kwargs):
        super(Sig5, self).__init__(activation, **kwargs)
        self.__name__ = 'sig5'

def sig5(x):
  return (keras.backend.sigmoid(x)*5)
get_custom_objects().update({'sig5': Sig5(sig5)}) 

def loadtrain(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  X = list()
  y = list()
  for i, row in enumerate(rows):
    if i != 0:
      y.append(np.array(row[3]))
      X.append(list(map(int, [row[1], row[2]])))  
  text.close()
  return np.array(X), np.array(y)

def loadtrainvari(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  X = list()
  y = list()
  allofusr = {}
  for i, row in enumerate(rows):
    if i != 0:
      y.append(np.array(row[3]))
      X.append(list(map(int, [row[1], row[2]])))
      try:
        allofusr[int(row[1])].append(int(row[3]))
      except:
        allofusr[int(row[1])] = [int(row[3])]
  text.close()
  for usr, scores in allofusr.items():
    allofusr[usr] = [np.mean(allofusr[usr]), np.std(allofusr[usr])]
  with open('./dict.pkl', 'wb') as dictflie:
    pickle.dump(allofusr, dictflie)
  new_y = []
  std = []
  for [usr, mov], score in zip(X, y):
    # p(allofusr[usr][0])
    # if allofusr[usr][1] == 0:
    #   r = 0
    # else:
    #   r = (score.astype('float') - float(allofusr[usr][0]))/float(allofusr[usr][1])
    # new_y.append(r)
    new_y.append(score)
    if float(allofusr[usr][1]) == 0:
      std.append(np.array([float(allofusr[usr][0]),1]))
    else:
      std.append(np.array([float(allofusr[usr][0]),float(allofusr[usr][1])]))
  # p(new_y)
  return np.array(X), np.array(new_y), np.array(std)

def loadtest(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  testX = list()
  for i, row in enumerate(rows):
    if i != 0:
      testX.append(list(map(int, [row[1], row[2]])))  
  text.close()
  return np.array(testX)

def loadtestvari(filedir):
  with open('./dict.pkl', 'rb') as dictflie:
    allofusr = pickle.load(dictflie)
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  testX = list()
  for i, row in enumerate(rows):
    if i != 0:
      testX.append(list(map(int, [row[1], row[2]])))  
  text.close()

  std = []
  for [usr, mov] in testX:
    if float(allofusr[usr][1]) == 0:
      std.append(np.array([float(allofusr[usr][0]),1]))
    else:
      std.append(np.array([float(allofusr[usr][0]),float(allofusr[usr][1])]))
  return np.array(testX), np.array(std)

def datasplit(oriX, oriy, rate):
    X = []
    y = []
    return X, y

def shuffle(shufflelist):
  r = shufflelist[0]
  for i, ele in enumerate(shufflelist):
    if i != 0:
      r = np.concatenate((r,ele), axis=1)
  np.random.shuffle(r)
  return r

def saveresult(savedir, result):
  data2write = [['TestDataID', 'Rating']]
  for i, row in enumerate(result):
    data2write.append([int(i+1), row[0]])
  text = open(savedir, "w+")
  s = csv.writer(text, delimiter=',',lineterminator='\n')
  for i in data2write:
      s.writerow(i) 
  text.close()

def loaduser(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  usrd = {}
  for i, row in enumerate(rows):
    if i != 0:
      row = row[0].split('::')
      if row[1] == 'F':
        sex = 0.
      else:
        sex = 1.
      zipcode = row[4].split('-')[0]
      usrd[float(row[0])] = [sex, float(row[2]), float(row[3]), float(zipcode)]
  text.close()
  return usrd

def loadmovie(filedir):
  import pandas as pd
  movie = pd.read_csv(filedir,sep="::",engine='python')
  tmp = []
  for topic in movie:
    tmp.append(movie[topic].values.tolist())
  movD = {}
  catNum = 0
  Allkind = []
  
  for idd, year, cat in zip(tmp[0], tmp[1], tmp[2]):
    year = float(year.split('(')[-1][:4])
    for kind in cat.split('|'):
      if Allkind.count(kind) == 0:
        Allkind.append(kind)
    cat = cat.split('|')
    movD[int(idd)] = [year,cat]

  for data in movD.items():
    # BOW = [0]*len(Allkind)
    # p(data[1][1])
    # for kind in data[1][1]:
    #   hit = Allkind.index(kind)
    #   BOW[hit] += 1
    movD[data[0]] = [data[1][0]] + [Allkind.index(data[1][1][0])]#BOW
  return movD

def p(m):
  print(m)
  exit()
