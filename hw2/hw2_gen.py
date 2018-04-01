import datainput as di
import Logisticmodel as lm
import sys
import numpy as np
import tools as tl
from tools import p

def predict(predictdir, outputdir, save, calerr):
  maxmin = np.load('./gen_maxmin.npy')
  maxscl = maxmin[0]
  minscl = maxmin[1]
  w = np.load('./gen_model.npy')
  print(w)
  # p(w)
  feature = np.load('./gen_feature.npy')
  # x2 = np.ndarray.tolist(np.load('./gen_x2.npy'))
  # x3 = np.ndarray.tolist(np.load('./gen_x3.npy'))
  # x4 = np.ndarray.tolist(np.load('./gen_x4.npy'))
  # x5 = np.ndarray.tolist(np.load('./gen_x5.npy'))
  # x6 = np.ndarray.tolist(np.load('./gen_x6.npy'))
  # x7 = np.ndarray.tolist(np.load('./gen_x7.npy'))
  # test = di.changefeature(test)
  # result = (-1)*test.dot(w)

  '''--------------calcutate result---------------------'''
  # test = di.selftest('./data/train_X')
  test = di.readtest(predictdir)
  # test = di.changefeature(test)
  # result = (-1)*test.dot(w)

  test = tl.scaling(test, maxscl, minscl)
  test = np.concatenate((np.ones((test.shape[0],1)),test), axis=1)
  test = tl.getfeature(feature,test)
  # test = tl.addxn(x2,test,2)
  # test = tl.addxn(x3,test,3)
  # test = tl.addxn(x4,test,4)
  # test = tl.addxn(x5,test,5)
  # test = tl.addxn(x6,test,6)
  # test = tl.addxn(x7,test,7)
  # test = test[:,1:]
  # p(test.shape)
  # b = w[:1]
  # w = w[1:]
  result = lm.sigmoid(w.dot(test.T))
  # result = tl.scaling(result, maxscl, minscl)
  result, hmax, hmin = tl.rescaling(result)
  for j in range(len(result)):
    if result[j] < 0.5:
      result[j] = 1
    else:
      result[j] = 0
  if save == True:
    di.output(outputdir,result)
  print('done')
  '''----------selftest-----------'''
  if calerr == 't':
    x = di.selftest('./data/train_X')
    x = tl.scaling(x, maxscl, minscl)
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    x = tl.getfeature(feature,x)
    vary = lm.sigmoid(w.dot(x.T))
    vary, hmax, hmin = tl.rescaling(vary)
    for i in range(len(vary)):
      if vary[i] < 0.5:
        vary[i] = 1
      else:
        vary[i] = 0
    y = di.readcsv('./data/train_Y',"y")
    err = lm.vaild(vary, y)
    print('error',err)
  
if __name__ == "__main__":
  if len(sys.argv) > 3:
    selftest = sys.argv[3]
  else:
    selftest = 'f'
  predict(sys.argv[1], sys.argv[2], True, selftest)