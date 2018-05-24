from time import time as now
start = now()
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

vec_dim = 200

if sys.argv[3] == 'public':
    model = load_model('./ensemble.h5')
elif sys.argv[3] == 'private':
    model = load_model('./ensemble.h5')
else:
    model = load_model(sys.argv[3])

# mo2 = load_model('model.0.0011-0.83055.h5')

# read data
test_X = readtest(sys.argv[1])
test_X = [ s for s in gensim.parsing.porter.PorterStemmer().stem_documents(test_X)]
Xlist_test = [ word.split(" ") for word in test_X]

# rawX = get_data(sys.argv[1])
# rawX, _ = get_t(sys.argv[1])
# Xlist = [ word.split(" ") for word in rawX]
print('got',len(Xlist_test),'data.')
w2v = Word2Vec.load('./w2v')
# word_seq, word_idx = tokenize(rawX)
with open('./tknzr.pickle', 'rb') as tknfile:
    tknzr = pickle.load(tknfile)

# print(w2v.wv["he"])

# for i in tknzr.word_index.items():
#   print(i)
word_seq = tknzr.texts_to_sequences(test_X)
word_idx = tknzr.word_index
print()
print('rawX[0]:', test_X[0])
print('word_seq[0]:', word_seq[0])
print('word_idx["wtf"]:', word_idx["wtf"])
# print(tknzr.word_idx)
wordNum = 39#len(word_idx)
print('got tokens:', len(word_idx))
embedding_weights = idx2weight(word_idx, vec_dim, w2v)

X = pad_sequences(word_seq, maxlen=39)

result = model.predict(X, batch_size=625)


data2write = [['id', 'label']]
for i, row in enumerate(result):
    # if row > 0.5:
    #     row = 1
    data2write.append([int(i), int(np.argmax(row))])
text = open(sys.argv[2], 'w+')
s = csv.writer(text, delimiter=',', lineterminator='\n')
for i in data2write:
    s.writerow(i) 
text.close()
print('use time', now() - start)