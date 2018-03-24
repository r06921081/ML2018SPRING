import datainput as di
import Logisticmodel as lm
import sys
import numpy as np
import tools as tl
def p(m):
  print(m)
  exit()

x = di.readcsv(sys.argv[1],"x")
# x, max, min = tl.rescaling(x)
y = di.readcsv(sys.argv[2],"y")
w = np.ones(x.shape[1])*0.5
w = lm.regression(x, y, w, 100, 10000)
print(w)

test = di.readtest(sys.argv[3])
# test = di.selftest(sys.argv[1])
# result = (-1)*test.dot(w)
result = 1/(1 + np.exp((-1)*test.dot(w)))
print(1/(1 + np.exp((-1)*test.dot(w))))
print(result)
for i in range(len(result)):
  if result[i] < 0.56:
    result[i] = 0
  else:
    result[i] = 1

di.output(sys.argv[4],result)
# p(np.sum(np.absolute(y - result)))

p(result)