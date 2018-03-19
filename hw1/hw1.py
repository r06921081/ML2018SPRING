
import numpy as np
import csv
import sys

X2_on = 0
def appendx2(X, x2select):
  X = np.concatenate((X,np.power(X[x2select,:],2)*0.01),axis=0)
  return X

def getfeature(dellist, X):
  tmp = np.ones((1, np.size(X[0,:])), dtype=np.float)  

  for ele in dellist:
    tmp = np.concatenate((tmp,X[ele,:]),axis=0)
  return tmp[1:]

w = np.load('model.npy')
chosce = np.ndarray.tolist(np.load('chosce.npy'))


test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")
# print(text)
n_row = 0
    
test_day = []
tmp_9hr = []
test_day = []
for r in row: # 把所有元素放在同一水平上好做feature selecttion
  tmp_eles = []
  if (n_row) % 18 == 0:
    if len(test_day) != 0:
      tmp_9hr = np.matrix(tmp_9hr,np.float)
      test_day = np.concatenate((test_day,tmp_9hr),axis=1)
    else:
      test_day = np.array(tmp_9hr)
    tmp_9hr = []
  for ele in range(2,11):
    if r[ele] != "NR":
      tmp_eles.append(r[ele])
    else:
      tmp_eles.append(0)
  tmp_9hr.append(tmp_eles)
  n_row += 1
test_day = np.concatenate((test_day,tmp_9hr),axis=1)
text.close()

test_day = np.matrix(test_day,np.float)

if X2_on == 2:
  for i in range(len(chosce)): # 把二次項的選擇加入chosce list 中給他選
    chosce.append(test_day[0].shape[0] + i) 
  for x2 in chosce:
    test_day = appendx2(test_day,x2)
elif X2_on == 1:
    chosce.append(18)
    test_day = appendx2(test_day,9)
test_day = getfeature(chosce,test_day)
final = []
for i in range(test_day[0].shape[1]//9):
  tmpX = test_day[:,i * 9:(i + 1) * 9]
  final.append(np.reshape(np.array(tmpX), 9*len(tmpX)))

final = np.array(final,dtype = np.float)
final = np.concatenate((np.ones((final.shape[0],1)),final), axis=1)
    
sol = final.dot(w.T)

out = []
for i in range(len(sol)):
  out.append(["id_"+str(i)])
  out[i].append(sol[i])

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(out)):
    s.writerow(out[i]) 
text.close()
print('done')