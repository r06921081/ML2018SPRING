import numpy as np
import sys
#import matplotlib.pyplot as plt
from sklearn.cluster import Birch, KMeans
from sklearn import cluster
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import os.path

from tools import p
import dataprocess as dp

def redo(traindir, testdir, savedir):
  X = np.load(traindir)
  X = X.reshape(X.shape[0], -1)/255.
  # X = X[:20000]
  print('X shape:', X.shape)
  test_x = dp.readtest(testdir)
  print('test_x shape:', test_x.shape)

  pca = PCA(n_components=410, whiten=True, svd_solver='full', random_state=0)
  ppa = pca.fit_transform(X) 

  # clustering 
  kmean= KMeans(n_clusters=2, random_state=33)
  kmean.fit(ppa)
  pt = kmean.fit_transform(ppa)

  result = []
  for row in test_x:
      if kmean.labels_[row[0]] == kmean.labels_[row[1]]:
          result.append(1)
      else:
          result.append(0)

  save(savedir, result, [pca, kmean])


def reproducetion(testdir, savedir):
  test_x = dp.readtest(testdir)
  print('test_x shape:', test_x.shape)
  with open('./pca_model.pickle', 'rb') as f:
    model = pickle.load(f)
    kmean = model[1]
  result = []
  for row in test_x:
      if kmean.labels_[row[0]] == kmean.labels_[row[1]]:
          result.append(1)
      else:
          result.append(0)
  save(savedir, result)

def save(savedir, result, models = None):
  dp.savepre(result, savedir)
  if models != None:
    with open('./pca_model.pickle', 'wb') as f:
      pickle.dump(models, f)

if __name__ == "__main__":
  paraNum = len(sys.argv)
  traindir = sys.argv[1] # './data/image.npy'
  testdir = sys.argv[2] # './data/test_case.csv'
  savedir = sys.argv[3] # './result/test.csv'
  if os.path.isfile('./pca_model.pickle'):
    print('pca_model.pickle exist, just reporduct.')
    reproducetion(testdir, savedir)
  else:
    print('pca_model.pickle not exist retrain.')
    redo(traindir, testdir, savedir)