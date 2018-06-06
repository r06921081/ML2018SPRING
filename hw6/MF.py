import os
import numpy as np
import sys
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.regularizers import l2
from utils import *
from keras.layers import Input, Embedding, Flatten, Add, Dot, Concatenate, Subtract, Multiply
from keras.models import Model
from keras import optimizers

batchsize = 1024
std = 1.116897661
mean = 3.581712086
outdim = 512

X, y, std = loadtrainvari(sys.argv[1])
# y = (y.astype(float) - mean)/std

pack = shuffle([X, y.reshape(-1,1), std])
X = pack[:,:2]
y = pack[:,2]
std = pack[:,3:]
std[:,1] = 1/std[:,1]
monitor = 'val_non_normalize_rmse'
callback = [
  ModelCheckpoint('./'+str(outdim)+'model.{'+monitor+':.4f}.h5', monitor=monitor, period=1),
  ReduceLROnPlateau(monitor=monitor, factor=0.64, patience=int(1), verbose=1),
  EarlyStopping(monitor=monitor, patience = 4)
  ]


usrdim = int(np.amax(X[:,0]))
movdim = int(np.amax(X[:,1]))

for i in range(50):
  usrIn = Input(shape = (1,))
  movIn = Input(shape = (1,))
  meanIn = Input(shape = (1,))
  stdIn = Input(shape = (1,))

  usrBi = usrIn
  movBi = movIn

  usrEmbedding = Embedding(
    input_dim = usrdim,
    output_dim = outdim,
    embeddings_initializer='he_uniform',
    input_length = 1)(usrIn)
  usr = Flatten()(usrEmbedding)

  movEmbedding = Embedding(
    input_dim = movdim, 
    output_dim = outdim, 
    embeddings_initializer='he_uniform',
    input_length = 1)(movIn)
  mov = Flatten()(movEmbedding)

  usrBiasEmbedding = Embedding(
    input_dim = usrdim, 
    output_dim = 1,
    trainable = True,
    embeddings_initializer='zero')(usrBi)
  usrBias = Flatten()(usrBiasEmbedding)

  movBiasEmbedding = Embedding(
    input_dim = movdim,
    output_dim = 1,
    trainable = True,
    embeddings_initializer='zero')(movBi)
  movBias = Flatten()(movBiasEmbedding)


  dot = Dot(axes = 1)([usr, mov])
  add = Add()([dot, usrBias, movBias])
  add = Subtract()([add, meanIn])
  add = Multiply()([add, stdIn])

  model = Model([usrIn, movIn, meanIn, stdIn], add)
  model.summary()

  opt = optimizers.Adam(lr=0.00064)
  model.compile(optimizer = opt, loss = 'mse', metrics = [non_normalize_rmse])
  history = model.fit(np.hsplit(X, 2) + np.hsplit(std, 2) , y,
            batch_size = batchsize, 
            epochs = 50, 
            validation_split=0.1,
            callbacks = callback)
  # from myplot import *
  # show_history(history, 'normalize_rmse', 'val_normalize_rmse')