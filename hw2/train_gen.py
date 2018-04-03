import datainput as di
import Logisticmodel as lm
import sys
import numpy as np
import tools as tl
from tools import p
import hw2_gen

feature = [-1]
x2 = [1,79,80,81]
x3 = [1,79,80,81]
x4 = [1,79,80,81]
x5 = [1,79,80]
x6 = [1,79,80]
x7 = [1,79,80]
# feature = [0,1,2,3,4,5,6,7,8,9,10,
#   11,12,13,14,15,16,17,18,19,20,
#   32,35,36,38,41,43,44,45,49,50,
#   51,52,53,54,56,57,58,59,60,61,
#   62,63,64,67,68,69,70,73,79,80,#71-80 *79
#   81,82,84,85,86,87,88,90,92,93,
#   94,96,97,98,99,106,107,120
#   ]
# x2 = [feature.index(1),feature.index(79),feature.index(80),feature.index(81)]
# x3 = [feature.index(1),feature.index(79),feature.index(80),feature.index(120)]
# x4 = [feature.index(1),feature.index(79),feature.index(80)]
# x5 = [feature.index(1),feature.index(79),feature.index(80)]
# x6 = [feature.index(1),feature.index(79),feature.index(80)]
# x7 = [feature.index(1),feature.index(79),feature.index(80)]
x = di.readcsv(sys.argv[1],"x")
x, max, min = tl.rescaling(x)
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)


x = tl.getfeature(feature,x)
x = np.array(x)
x = tl.addxn(x2,x,2)
x = tl.addxn(x3,x,3)
# x = tl.addxn(x4,x,4)
# x = tl.addxn(x5,x,5)
# x = tl.addxn(x6,x,6)
# x = tl.addxn(x7,x,7)
x = x[:,1:]

y = di.readcsv(sys.argv[2],"y")

# ss = np.ones((x.shape[1],1))

# data_num = x.shape[0]
# fea_num = x.shape[1]#np.identity(fea_num)#
# f = np.concatenate((ss, np.full([x.shape[1], x.shape[1]-1], 0)), axis = 1).T
# # f[1,0] = data_num
# # f = np.tile(f,(100,1,1))


# # x = x.reshape(100,fea_num,1)

# w = np.tile(w,(data_num,1,1))
# ppp=np.multiply(np.dot(x,f),x)
# ans = x[0,:].T.dot(f).dot(x[0,:])
# print(x)
# print(np.sum(ppp,axis=1))
# print(ans)
# p(np.sum(ppp,axis=1).shape)
# p(np.dot(x,f).shape)
w, b = lm.Generative(x, y, 100, 5000)
wtmp = np.array(b).flatten()
for i in range(w.shape[1]):
  wtmp = np.concatenate((wtmp,[w[0,i]]),axis = 0)
np.save('./gen_model.npy',wtmp)
np.save('./gen_feature.npy',feature)
np.save('./gen_maxmin.npy', [max, min])
np.save('./gen_x2.npy',x2)
np.save('./gen_x3.npy',x3)
# np.save('./gen_x4.npy',x4)
# np.save('./gen_x5.npy',x5)
# np.save('./gen_x6.npy',x6)
# np.save('./gen_x7.npy',x7)

print(w)

hw2_gen.predict(sys.argv[1],sys.argv[4],True,'t')
p('done')
