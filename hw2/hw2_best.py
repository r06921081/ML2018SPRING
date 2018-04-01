import datainput as di
import Logisticmodel as lm
import sys
import numpy as np
import tools as tl
from tools import p

def predict(predictdir, outputdir, power, save, calerr):
  maxmin = np.load('./best_maxmin.npy')
  maxscl = maxmin[0]
  minscl = maxmin[1]
  w = np.load('./best_model.npy')
  print(w)
  # p(w)
  feature = np.load('./best_feature.npy')
  powarr = []
  for i in range(1,power):
    powarr.append(np.ndarray.tolist(np.load('./best_x' + str(i + 1) + '.npy')))

  # test = di.selftest('./data/train_X')
  test = di.readtest(predictdir)
  # test = di.changefeature(test)
  # result = (-1)*test.dot(w)

  test = tl.scaling(test, maxscl, minscl)
  test = np.concatenate((np.ones((test.shape[0],1)),test), axis=1)
  test = tl.getfeature(feature,test)
  for i in range(1,power):
    test = tl.addxn(powarr[i - 1], test, i + 1)
  # test = tl.addxn(x3,test,3)
  # test = tl.addxn(x4,test,4)
  # test = tl.addxn(x5,test,5)
  # test = tl.addxn(x6,test,6)
  # test = tl.addxn(x7,test,7)
  result = lm.sigmoid(test.dot(w))
  result, hmax, hmin = tl.rescaling(result)
  for i in range(len(result)):
    if result[i] < 0.5:
      result[i] = 0
    else:
      result[i] = 1
  if save == True:
    di.output(outputdir,result)
  print('done')
  '''----------selftest-----------'''
  if calerr == 't':
    x = di.selftest('./data/train_X')
    x = tl.scaling(x, maxscl, minscl)
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    x = tl.getfeature(feature,x)
    for i in range(1,power):
      x = tl.addxn(powarr[i - 1], x, i + 1)
    vary = lm.sigmoid(x.dot(w))
    vary, hmax, hmin = tl.rescaling(vary)
    for i in range(len(vary)):
      if vary[i] < 0.5:
        vary[i] = 0
      else:
        vary[i] = 1
    y = di.readcsv('./data/train_Y',"y")
    err = lm.vaild(vary, y)
    print('error',err)
  
if __name__ == "__main__":
  if len(sys.argv) > 3:
    selftest = sys.argv[3]
  else:
    selftest = 'f'
  predict(sys.argv[1], sys.argv[2], 7, True, selftest)