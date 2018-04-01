import datainput as di
import Logisticmodel as lm
import sys
import numpy as np
import tools as tl
import hw2_logi

# feature = [0,1,2,3,4,5,6,7,8,9,
#   11,12,13,14,15,16,17,18,19,20,
#   32,35,36,38,41,43,44,45,49,50,
#   51,52,53,54,56,57,58,59,60,61,
#   62,63,64,67,68,69,70,73,79,80,#71-80 *79
#   81,82,84,85,86,87,88,90,92,93,
#   94,96,97,98,99,106,107,120
#   ]
# x2 = [feature.index(1),feature.index(79),feature.index(80),feature.index(81)]
feature = [-1]
x = di.readcsv(sys.argv[1],"x")
x, tmax, tmin = tl.rescaling(x)
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

x = tl.getfeature(feature,x)
x = np.array(x)
# x = tl.addxn(x2,x, 2)
# p(x)
y = di.readcsv(sys.argv[2],"y")
b = np.ones(1)
x = x[:,1:]
w = np.ones(x.shape[1])
w, b = lm.regression(x, y, w, b, 0.0, 0.001, 100000)
w = np.concatenate((b, w), axis = 0)
np.save('./logi_model.npy',w)
np.save('./logi_feature.npy',feature)
np.save('./logi_maxmin.npy', [tmax, tmin])
# np.save('./logi_x2.npy',x2)

print('w:', w)

hw2_logi.predict(sys.argv[3],sys.argv[4], True, 't')
print('done')
