import datainput as di
import Logisticmodel as lm
import sys
import numpy as np
import tools as tl
import hw2_best
def p(m):
  print(m)
  exit()
# feature = [0,1,2,3,4,5,6,7,8,9,
#   11,12,13,14,15,16,17,18,19,20,
#   32,35,36,38,41,43,44,45,49,50,
#   51,52,53,54,56,57,58,59,60,61,
#   62,63,64,67,68,69,70,73,79,80,#71-80 *79
#   81,82,83,84,85,86,87,88,89,90,
#   91,92,93,94,95,96,97,98,99,102,
#   106,107,120
#   ]
# x2 = [feature.index(1),feature.index(79),feature.index(80),feature.index(81)]
# x3 = [feature.index(1),feature.index(79),feature.index(80),feature.index(120)]
# x4 = [feature.index(1),feature.index(79),feature.index(80)]
# x5 = [feature.index(1),feature.index(79),feature.index(80)]
# x6 = [feature.index(1),feature.index(79),feature.index(80)]
# x7 = [feature.index(1),feature.index(79),feature.index(80)]
feature = [-1]
x2 = [1,79,80,81]
x3 = [1,79,80,120]
# x4 = [1,79,80]
# x5 = [1,79,80]
# x6 = [1,79,80]
# x7 = [1,79,80]
xarray = []
xarray.append(x2)
xarray.append(x3)
# xarray.append(x4)
# xarray.append(x5)
# xarray.append(x6)
# xarray.append(x7)
x = di.readcsv(sys.argv[1],"x")
# x = di.changefeature(x)

x, tmax, tmin = tl.rescaling(x)
  
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
x = tl.getfeature(feature,x)
for i in range(len(xarray)):
  x = tl.addxn(xarray[i],x,i + 2)
x = np.array(x)

# p(x)
y = di.readcsv(sys.argv[2],"y")

# tmpx = np.expand_dims(x[0,:], axis=0)
# tmpy = np.expand_dims(y[0], axis=0)
# for datax,datay in zip(x,y):
#   tmpx = np.concatenate((tmpx,np.tile(datax,(int(datax[10]*100//1),1))), axis=0)
#   tmpy = np.concatenate((tmpy,np.tile(datay,(int(datax[10]*100//1)))), axis=0)
# x = np.concatenate((x,tmpx), axis=0)
# y = np.concatenate((y,tmpy), axis=0)
# x,y = tl.shuffle(x,y)

w = np.ones(x.shape[1])
w,y_array = lm.regression_p(x, y, w, 0, 0.0, 65,3000000)
np.save('./best_model.npy',w)
np.save('./best_feature.npy',feature)
np.save('./best_maxmin.npy', [tmax, tmin])
for i in range(len(xarray)):
  np.save('./best_x' + str(i + 2) + '.npy',xarray[i])
# np.save('./best_x3.npy',x3)
# np.save('./best_x4.npy',x4)
# np.save('./best_x5.npy',x5)
# np.save('./best_x6.npy',x6)
# np.save('./best_x7.npy',x7)

print(w)

hw2_best.predict(sys.argv[3],sys.argv[4], len(xarray) + 1, True, 't')