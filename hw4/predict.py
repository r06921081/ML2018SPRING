import numpy as np
import sys
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
# import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn import cluster
from sklearn.manifold import Isomap
from tools import p
import dataprocess as dp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd

'''
argv[1] train data
argv[2] test data
argv[3] result dir
'''

X = np.load(sys.argv[1])

# X = X - np.mean(X)
X = X/255.
image = X
# x_val = X[:val_part].reshape((val_part, 28, 28, 1))
model = load_model('./new.h5')
model.summary()

encoder = Model(inputs=model.get_layer('input_1').input,
                                 outputs=model.get_layer('d3').output)

cnn_input = Input(model.get_layer('d4').input_shape[1:])

X_autoencoder = encoder.predict(image.reshape(-1,28,28,1))

test_x = dp.readtest(sys.argv[2])

# pca300 = PCA(n_components=100, whiten=True, svd_solver='full',random_state=0)
# pca300.fit(X_autoencoder)
# ppa300 = pca300.fit_transform(X_autoencoder) 


kmean = KMeans(n_clusters=2, n_jobs=12, random_state=0)
kmean.fit(X_autoencoder)

def get_result(data):
    result = []
    for row in test_x:
      if data.labels_[row[0]] == data.labels_[row[1]]:
          result.append(1)
      else:
          result.append(0)
    return result

dp.savepre(get_result(kmean), sys.argv[3])
p('9')