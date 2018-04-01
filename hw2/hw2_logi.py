import datainput as di
import Logisticmodel as lm
import sys
import numpy as np
import tools as tl
from tools import p

def predict(predictdir, outputdir, save, calerr):
  maxmin = np.load('./logi_maxmin.npy')
  maxscl = maxmin[0]
  minscl = maxmin[1]
  w = np.load('./logi_model.npy')
  b = w[:1]
  w = w[1:]
  feature = np.load('./logi_feature.npy')
  # x2 = np.ndarray.tolist(np.load('./logi_x2.npy'))
  
  
  test = di.readtest(predictdir)

  # test = di.changefeature(test)
  # result = (-1)*test.dot(w)

  '''--------------calcutate result---------------------'''
  test = tl.scaling(test, maxscl, minscl)
  test = np.concatenate((np.ones((test.shape[0],1)),test), axis=1)
  test = tl.getfeature(feature,test)
  test = test[:,1:]
  # test = tl.addxn(x2,test,2)
  result = lm.sigmoid(test.dot(w) + b)
  # result = tl.scaling(result, maxscl, minscl)
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
    x = x[:,1:]
    vary = lm.sigmoid(x.dot(w) + b)
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
  predict(sys.argv[1], sys.argv[2], True, selftest)