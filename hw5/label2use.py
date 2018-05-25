from keras.models import load_model
import numpy as np
import sys
import csv
import gensim
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, Input, Embedding
import keras as K
import re
import pickle
from utils import *
from time import time as now
start = now()
vec_dim = 200
# load mode
# def swish(x):
#   return (K.backend.sigmoid(x)*x)
# get_custom_objects().update({'swish': Activation(swish)}) 
if sys.argv[3] == 'public':
    model = load_model('./')
elif sys.argv[3] == 'private':
    model = load_model('./')
else:
    model = load_model(sys.argv[3])

# mo2 = load_model('model.0.0011-0.83055.h5')
test = False
numofmodel = int(sys.argv[4])
# read data
if test:
    semi_X = readtest(sys.argv[1])
else:
    semi_X, _ = get_nolabel(sys.argv[1])
semi_X = [ s for s in gensim.parsing.porter.PorterStemmer().stem_documents(semi_X)]
Xlist_semi = [ word.split(" ") for word in semi_X]

# p(semi_X[0:20])
# rawX = get_data(sys.argv[1])
# rawX, _ = get_t(sys.argv[1])
# Xlist = [ word.split(" ") for word in rawX]
print('got',len(Xlist_semi),'data.')
w2v = Word2Vec.load('./w2v')
# word_seq, word_idx = tokenize(rawX)
with open('./tknzr.pickle', 'rb') as tknfile:
    tknzr = pickle.load(tknfile)

# print(w2v.wv["he"])

# for i in tknzr.word_index.items():
#   print(i)
word_seq = tknzr.texts_to_sequences(semi_X)
word_idx = tknzr.word_index
print()
print('rawX[0]:', semi_X[0])
print('word_seq[0]:', word_seq[0])
print('word_idx["wtf"]:', word_idx["wtf"])
# print(tknzr.word_idx)
wordNum = 39#len(word_idx)
print('got tokens:', len(word_idx))
embedding_weights = idx2weight(word_idx, vec_dim, w2v)

X = pad_sequences(word_seq, maxlen=39)

result = model.predict(X, batch_size = 512)
result = result/numofmodel
print('use time', now() - start)
data2write = []
for i, row in enumerate(result):
    if 1 > row[0] and row[0] > 0.65:
        t = [str(0) + '+$+' + semi_X[i]]
        data2write.append([str(0) + '+$+' + semi_X[i]])
        # print(row)
    elif 1 > row[1] and row[1] > 0.65:
        t = [str(1) + '+$+' + semi_X[i]]
        data2write.append([str(1) + '+$+' + semi_X[i]])
        # print(row)
    # print(np.sum(row))
text = open(sys.argv[2], 'w+')
# s = csv.writer(text, delimiter=',', lineterminator='\n')
for i in data2write:
    # string = i[0]
    # print(string)
    text.write(i[0]+'\n')
    # s.writerow([string]) 
text.close()
