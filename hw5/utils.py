import os
import numpy as np
import re
import csv
import keras as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
quk = 10000000
def p(m):
  print(m)
  exit()

def replacerule(match_obj):
  # if a char 'c' continue over twice then make it 'cc'
  return match_obj.group(0)[0]

def tokenize(data, forceWrite=False):
  from keras.preprocessing.text import Tokenizer
  import pickle
  """
  get the sequence num of each sentence
  get the all word index number
  """
  tknzr = Tokenizer(filters='\t',)
  tknzr.fit_on_texts(data)

  if not os.path.exists('./tknzr.pickle') or forceWrite:
    with open('tknzr.pickle', 'wb') as tknfile:
      pickle.dump(tknzr, tknfile)
  else:
    print('file ./tknzr.pickle exists pass!')
    with open('./tknzr.pickle', 'rb') as tknfile:
      tknzr = pickle.load(tknfile)

  return tknzr.texts_to_sequences(data), tknzr.word_index

class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
  return (K.backend.sigmoid(x)*x)
get_custom_objects().update({'swish': Swish(swish)}) 

class hardSwish(Activation):
    def __init__(self, activation, **kwargs):
        super(hardSwish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def hard_swish(x):
  return (K.backend.hard_sigmoid(x)*x)
get_custom_objects().update({'hard_swish': hardSwish(hard_swish)}) 

def wordmanytime(match_obj):
  # if a char 'c' continue over twice then make it 'cc'
  # print(match_obj)
  if match_obj.group(0)[0] == match_obj.group(0)[1]:
    return match_obj.group(0)[0] 
  else:
    return match_obj.group(0)[0] + match_obj.group(0)[1]

def cleanData(rawdata):
  # replacelist = ['~', '@', '#', '$', '%', '^', 
  #                 '&', '*', '(', ')', '_', '+', '`',
  #                 '-', '=', '[', ']', '{', '}', '\\',
  #                 '|', '\'', '\"', #',', '.','?', '!', ';', ':'
  #                 '/', '<', '>' ]
  cleaned = []
  for row in rawdata:
    row = row.replace(" ' ", "'")
    # row = row.replace("+++$+++ ", "")
    row = row.replace("too", "to0")
    row = row.replace(" .. ", " .0. ")
    row = row.replace(" ... ", " .0. ")
    # test = test.replace("too", "to0")
    # print(test)
    # for si in replacelist:
    #   row = row.replace(si, "")
    row = re.sub(r'(((.){1,5})\2{1,})', wordmanytime, row, flags=re.IGNORECASE)
    row = re.sub(r'(((.){1,2})\2{1,})', wordmanytime, row, flags=re.IGNORECASE)
    # test = re.sub(r'(((.){1,2})\2{1,})', wordmanytime, test, flags=re.IGNORECASE)
    # print(test)
    # test = test.replace("to0", "too")
    # print(test)
    # exit()
    row = row.replace("to0", "too")
    row = row.replace(" .0. ", " ... ")
    cleaned.append(row)
  return cleaned

def get_data(datadir):
  test = "im sssssorrrrrrrrrrrry !!!!???@@1,,,...// too hahaha to . hehheh i think it was me that got you sick .... do u still love me"
  # test = re.sub(r'.{2,}',r'0' ,test)
  y = []
  rawX = []
  with open(datadir, newline='') as csvfile:
    raw = csv.reader(csvfile, delimiter='\n', quotechar=' ')
    for i, row in enumerate(raw):
      y.append(row[0].split(" +++$+++ ")[0])
      rawX.append(row[0].split(" +++$+++ ")[1])
      if i == quk-1:
        break
    rawX = cleanData(rawX)
    print(cleanData([test]))      
  print(rawX[0],y[0])
  return rawX, y

def get_nolabel(nolabledir):
  nolable_X = []
  index = []
  first = ''
  with open(nolabledir, newline='') as csvfile:
    raw = csv.reader(csvfile, delimiter='\n', quotechar=' ')
    for i, row in enumerate(raw):
      if i == 0:
        first = row[0]
      if len(row) != 1:
        nolable_X.append(first)
      else:
        nolable_X.append(row[0])
      index.append(i)
      if i == quk-1:
        break
  # print(nolable_X[0][0])
  return cleanData(nolable_X), index

def get_semidata(datadir):
  # test = re.sub(r'.{2,}',r'0' ,test)
  y = []
  semiX = []
  with open(datadir, newline='') as csvfile:
    raw_semi = csv.reader(csvfile, delimiter='\n', quotechar=' ')
    for i, row in enumerate(raw_semi):
      y.append(row[0].split("+$+")[0])
      semiX.append(row[0].split("+$+")[1])
      if i == quk-1:
        break
  print(semiX[0],y[0])
  return semiX, y

def readtest(testdir):
  test_X = []
  with open(testdir, newline='') as csvfile:
    raw = csv.reader(csvfile, delimiter='\n', quotechar=' ')
    for i, row in enumerate(raw):
      if i == 0:
        continue
      # elif i == 1:
      #   print(row)
      tmplist = row[0].split(',')

      if len(tmplist) > 2:
        tmplist = ','.join(tmplist[1:])
        test_X.append(tmplist)
      else:
        test_X.append(tmplist[1])
      if quk - 1 == i:
        break
  return cleanData(test_X)

def idx2weight(word_idx, vec_dim, w2v):  
  unknow = 0
  weight_arr = np.zeros((len(word_idx) + 1, vec_dim)) # (all the word number)*(w2v dim)
  get = 0
  for word, i in word_idx.items(): # translate all the (word index) -> w2v
    try:
      weight_arr[i] = w2v.wv[word]
      get += 1
    except:
      unknow +=1
  print('unknow word:', unknow, ', get:', get)
  return weight_arr

def embedding(listdata, dim = 300, window = 5, forceWrite = False):
  from gensim.models import Word2Vec
  if os.path.exists('./w2v') and forceWrite == False:
    print('file ./w2v exists pass!')
    return Word2Vec.load('./w2v')

  emb_model = Word2Vec(listdata, size=dim, window=window, min_count=5, workers=8, sg=1, seed=0)
  print('similar to dog', emb_model.most_similar("dog"))
  print('embeding dim:', len(emb_model["dog"]))
  print('vocab size:', len(emb_model.wv.vocab))
  emb_model.save('./w2v')
  return emb_model

def datasplit(X, y, partNum, total):
  frac = partNum/total
  dataNum = len(X)
  trainX = np.concatenate((X[0:partNum*dataNum//total],X[(partNum+1)*dataNum//total:dataNum]), axis=0)
  validX = X[partNum*dataNum//total:(partNum+1)*dataNum//total]
  trainy = np.concatenate((y[0:partNum*dataNum//total],y[(partNum+1)*dataNum//total:dataNum]), axis=0)
  validy = y[partNum*dataNum//total:(partNum+1)*dataNum//total]
  return trainX, validX, trainy, validy



if __name__ == '__main__':
  import sys
  import gensim
  rawX, y = get_data('./data/training_label.txt')
  rawX = [ s for s in gensim.parsing.porter.PorterStemmer().stem_documents(rawX)]
  Xlist = [ word.split(" ") for word in rawX]
  embedding(Xlist, 100, 10)