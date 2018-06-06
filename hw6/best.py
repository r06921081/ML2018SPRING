import os
import numpy as np
import sys
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.regularizers import l2
from utils import *
from keras.layers import Input, Embedding, Flatten, Add, Dot, Dense, Concatenate, Dropout, Multiply, Conv1D, Subtract, Lambda, BatchNormalization
from keras.models import Model
from keras import optimizers
import keras.backend as K
from keras.regularizers import l2, l1

batchsize = 1024
std = 1.116897661
mean = 3.581712086
outdim = 512
lr = 0.000641
lossfunc = 'normalize_rmse'

X ,y, stdtmp = loadtrainvari(sys.argv[1])
personmean = stdtmp[:,0] - mean
personstd = 1/stdtmp[:,1]
usrdim = 6040#int(np.amax(X[:,0],axis=0))
movdim = 4500#int(np.amax(X[:,1],axis=0))

y = (y.astype(float) - mean)/std
# y[:,0] = (y[:,0] - y[:,1])/std


movie = loadmovie(sys.argv[2])
user = loaduser(sys.argv[3])
# p(movie)
# p(user)
X_movieYear = np.array([movie[movieId][0] for movieId in X[:,1]])
X_movieType = np.array([movie[movieId][1:] for movieId in X[:,1]])
X_user = np.array([user[userId] for userId in X[:,0]])

pack = shuffle([X, y.reshape(-1,1),X_movieYear.reshape(-1,1),X_movieType,X_user])
X = pack[:,:2]
y = pack[:,2]
X_movieYear = pack[:,3] 
X_movieType = pack[:,4]

countYear = {}
for i in X_movieYear:
  countYear[i] = 1

yeardim = len(countYear.items())

callback = [
  ModelCheckpoint('./'+str(outdim)+'Bmodel.{val_'+lossfunc+':.4f}.h5', monitor='val_'+lossfunc, period=1),
  ReduceLROnPlateau(monitor='val_'+lossfunc, factor=0.64, patience=int(1), verbose=1),
  EarlyStopping(monitor='val_'+lossfunc, patience = 8)
  ]

while True:
  usrIn = Input(shape = (1,))
  movIn = Input(shape = (1,))
  yearIn = Input(shape = (1,))
  TypeIn = Input(shape = (1,))

  usrBi = usrIn
  movBi = movIn

  usrEmbedding = Embedding(
    input_dim = usrdim,
    output_dim = outdim,
    # trainable = True,
    # embeddings_regularizer=l1(0.00001),
    embeddings_initializer='he_uniform',
    input_length = 1)(usrIn)
  usr = Flatten()(usrEmbedding)

  movEmbedding = Embedding(
    input_dim = movdim, 
    output_dim = outdim, 
    # trainable = True,
    # embeddings_regularizer=l1(0.00001),
    embeddings_initializer='he_uniform',
    input_length = 1)(movIn)
  mov = Flatten()(movEmbedding)

  yearEmbedding = Embedding(
    input_dim = yeardim,
    output_dim = 1,
    # trainable = True,
    embeddings_initializer='he_uniform',
    input_length = 1)(yearIn)
  year = Flatten()(yearEmbedding)
  # year = Dropout(0.2)(year)
  # year = Dropout(0.5)(year)
  # year = Dense(outdim, activation="tanh")(yearEmbedding)

  TypeEmbedding = Embedding(
    input_dim = 18,
    output_dim = outdim,
    trainable = True,
    # embeddings_regularizer = l2(0.005),
    embeddings_initializer='he_uniform',
    input_length = 1)(TypeIn)
  # Type = Conv1D(64,3)(TypeEmbedding)
  # Type = Conv1D(128,3)(Type)
  Type = Flatten()(TypeEmbedding)
  Type = Dropout(0.1)(Type)
  # Type = Dense(outdim, activation="tanh")(Type)
  # Type = Dense(8, activation="tanh")(Type)
  # Type = Dropout(0.2)(Type)
  

  madd = Multiply()([mov, Type])
  # uadd = Dot(axes = 1)([usr, user])

  usrBiasEmbedding = Embedding(
    input_dim = usrdim, 
    output_dim = 1,
    trainable = True,
    # embeddings_regularizer=l2(0.00001),
    embeddings_initializer='zero')(usrBi)
  usrBias = Flatten()(usrBiasEmbedding)

  movBiasEmbedding = Embedding(
    input_dim = movdim,
    output_dim = 1,
    trainable = True,
    # embeddings_regularizer=l2(0.00001),
    embeddings_initializer='zero')(movBi)
  movBias = Flatten()(movBiasEmbedding)

  # usr = Multiply()([usr,user])
  dot = Dot(axes = 1)([usr, madd])
  add = Add()([dot, usrBias, movBias, year])
  # add = Lambda(lambda x: x * K.constant(std, dtype=K.floatx()))(add)
  # add = Lambda(lambda x: x + K.constant(mean, dtype=K.floatx()))(add)
  model = Model([usrIn, movIn, yearIn, TypeIn], add)
  model.summary()

  opt = optimizers.Adam(lr=lr)
  model.compile(optimizer = opt, loss = 'mse', metrics = [normalize_rmse])
  
  history = model.fit(np.hsplit(X, 2) + [X_movieYear, X_movieType], y,
            batch_size = batchsize, 
            epochs = 1000, 
            validation_split=0.05,
            callbacks = callback)
  # from myplot import *
  # show_history(history, 'normalize_rmse', 'val_normalize_rmse')