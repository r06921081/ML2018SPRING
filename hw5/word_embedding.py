
from keras.layers.wrappers import Bidirectional
from keras import layers
import numpy as np
# sentences list
import re
import csv
import sys
import os
import pickle
import gensim
from keras.models import load_model, Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, History ,TensorBoard, CSVLogger, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
import keras as K
from utils import *
from sklearn.model_selection import train_test_split
#-----------
from keras.utils import to_categorical
from keras.layers import Embedding, MaxPool1D,Input,InputLayer,BatchNormalization, Dense, Bidirectional,LSTM,Dropout,GRU,Activation, LeakyReLU, Conv1D, Flatten
from keras.utils.generic_utils import get_custom_objects
from keras import regularizers

#---------

if __name__ == '__main__':  
  vec_dim = 200
  window = 10
  batch_size=1300
  iterNum = 20
  forceWrite = False
  semi = True
  rawX, y = get_data(sys.argv[1])
  rawX = [ s for s in gensim.parsing.porter.PorterStemmer().stem_documents(rawX)]
  Xlist = [ word.split(" ") for word in rawX]

  semiX = []
  semiy = []
  Xlist_semi = []
  if semi and os.path.exists('./data/labeled.txt'):
    semiX, semiy = get_semidata('./data/labeled.txt')
    semiX_test, semiy_test = [], []#get_semidata('./data/labeled_test.txt')
    semiX = semiX + semiX_test
    semiy = semiy + semiy_test
    Xlist_semi = [ word.split(" ") for word in semiX]

  makeEmbedding = []
  makeToken = []
  if forceWrite and len(sys.argv) > 2:
    nolableX, index = get_nolabel(sys.argv[2])
    nolableX = [ s for s in gensim.parsing.porter.PorterStemmer().stem_documents(nolableX)]
    Xlist_nolabel = [ word.split(" ") for word in nolableX]

    # test_X = readtest('./data/testing_data.txt')
    # test_X = [ s for s in gensim.parsing.porter.PorterStemmer().stem_documents(test_X)]
    # Xlist_test = [ word.split(" ") for word in test_X]
    makeEmbedding = Xlist_nolabel# + Xlist_test
    makeToken = nolableX# + test_X
  
  w2v = embedding(Xlist + makeEmbedding, vec_dim, window, forceWrite)

  word_seq, word_idx = tokenize(rawX + semiX, forceWrite)

  print(len(word_seq))
  word_seq_semi = word_seq[len(rawX):]
  word_seq = word_seq[:len(rawX)]
  print([len(word_seq),len(word_seq_semi)])

  print('rawX[0]:', rawX[0])
  print('word_seq[0]:', word_seq[0])
  # print('word_seq_semi[0]:', word_seq_semi[0])
  print('word_idx["wtf"]:', word_idx["wtf"])
  # print(tknzr.word_idx)
  wordNum = len(word_idx)
  print('got tokens:', wordNum)
  embedding_weights = idx2weight(word_idx, vec_dim, w2v)

  # get longest sentence lenght
  longestSent = 0
  for i in word_seq:
    if len(i) > longestSent:
      longestSent = len(i)

  print("longest sentence:", longestSent)
  longestSent = 39
  # 
  X = pad_sequences(word_seq, maxlen=longestSent)
  X_semi = pad_sequences(word_seq_semi, maxlen=longestSent)

  # print(train_X_num.shape)
  print('word_seq[0]:', X[0])
  y_type = 'y_cat'
  if y_type == 'y_sigm':
    y = np.array(y, dtype=int) # sigmoid
    semiy = np.array(semiy, dtype=int)
  else:
    y = to_categorical(np.asarray(y)) # softmax
    semiy = to_categorical(np.asarray(semiy))
  print(y.shape)
 
  for i in range(iterNum):
    if i in [0,1,2,3,4,5,6]:
      continue
    callback = [
            TensorBoard(),
            CSVLogger('./log.csv', append=True),
            History(),
            ModelCheckpoint('./model.'+str(i)+'.{val_acc:.5f}-{val_loss:.5f}.h5', monitor='val_acc', period=1,save_best_only=False),
            ReduceLROnPlateau('val_acc', factor=0.780, patience=int(1), verbose=1),
            EarlyStopping(patience = 8, monitor='val_acc')
            ]
    # train_X, valid_X, train_y, valid_y = datasplit(X, y, i, iterNum)
    # train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.70, shuffle=True)
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.1, shuffle=True)
    if semi:
      # print(type(train_X))
      train_X = np.concatenate((train_X, X_semi), axis=0)
      train_y = np.concatenate((train_y, semiy), axis=0)
      # train_X = X_semi
      # train_y = semiy
    kernel_init = 'he_uniform'
    model = Sequential()
    model.add(Embedding(wordNum+1,
                        output_dim = vec_dim,
                        weights=[embedding_weights],
                        input_length=longestSent,
                        trainable=False))
    model.add(Bidirectional(GRU(256,
                activation='tanh',
                return_sequences = True,
                dropout = 0.2,
                recurrent_dropout = 0.4,
                # activity_regularizer=regularizers.l2(0.00001),
                kernel_initializer=kernel_init)))
    # model.add(BatchNormalization())
    model.add(Bidirectional(GRU(256,
                activation='tanh',
                return_sequences = False,
                dropout = 0.25,
                recurrent_dropout = 0.5,
                # activity_regularizer=regularizers.l2(0.0001),
                kernel_initializer=kernel_init)))
    # model.add(BatchNormalization())
    # model.add(Conv1D(filters=64, kernel_size=10, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dropout(0.35))
    # model.add(MaxPool1D(30))    
    
    # model.add(Flatten())
    model.add(Dense(256, activation=swish))
    # model.add(LeakyReLU(0.05))
    model.add(Dropout(0.5))
    # model.add(Dense(256))
    # model.add(LeakyReLU(0.05))
    # model.add(Dropout(0.12))
    # model.add(Dense(512))
    # model.add(LeakyReLU(0.05))
    # model.add(Dropout(0.15))
    # model.add(Dense(512))
    # model.add(LeakyReLU(0.05))
    # model.add(Dropout(0.20))
    # model.add(Dense(1024))
    # model.add(LeakyReLU(0.05))
    # model.add(Dropout(0.25))
    model.add(Dense(256, activation=swish))
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='softmax'))
    adam = optimizers.Adam(0.0015)# 'adam'
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    print('start train....')
    model.summary()
    # p([train_X.shape,train_y.shape,valid_X.shape, valid_y.shape])
    model.fit(train_X, train_y, validation_data=(valid_X, valid_y),
            epochs=40, batch_size=batch_size,callbacks=callback)
    model.save('./model.'+str(i)+'.{val_acc:.5f}.h5')
  