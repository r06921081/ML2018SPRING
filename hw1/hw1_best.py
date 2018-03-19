
import numpy as np
import csv
import sys

def appendx2(X, x2select):
  # print(X[9,:])
  X = np.concatenate((X,np.power(X[x2select,:],2)),axis=0)
  # print(X[18,:])
  return X

def getfeature(dellist, X):
  tmp = np.ones((1, np.size(X[0,:])), dtype=np.float)  

  for ele in dellist:
    tmp = np.concatenate((tmp,X[ele,:]),axis=0)
  return tmp[1:]

w = np.load('model_best.npy')
chosce = np.ndarray.tolist(np.load('chose_best.npy'))
print('lllllllllllllllllll')
print(w)
print('llllllllllllllllllll')
test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")
# print(text)

n_row = 0
# day_tmp = []
# for r in row:
#   if n_row % 18 == 0:
#     for l in day_tmp:
#       test_x.append(l)
#       day_tmp = []
#     for c in range(9):
#       day_tmp.append([str(n_row//18 + (c+1)/24)])
#   for ele in range(2,11):
#     if r[ele] != "NR":
#       day_tmp[ele - 2].append(r[ele])
#     else:
#       day_tmp[ele - 2].append(0)
#     # for ele in range(2,11):
#     #   day_tmp[ele - 2].append(r[ele])
#   # print(day_tmp)
#   # print(r)
#   n_row += 1

# for l in day_tmp:
#   test_x.append(l)
    
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
print(test_day.shape)
# for ele in chosce:
#   test_day = appendx2(test_day,ele)
test_day = appendx2(test_day,9)

test_day = getfeature(chosce,test_day)
print(test_day.shape)
final = []
for i in range(test_day[0].shape[1]//9):
  # print(i)
  tmpX = test_day[:,i * 9:(i + 1) * 9]
  final.append(np.reshape(np.array(tmpX), 9*len(tmpX)))
  # print(np.reshape(np.array(tmpX), 9*len(tmpX)))
  # if len(final) != 0:
  #   final = np.concatenate((np.array([[1]]),resh),axis=1))
  # else:
  #   final = np.array(tmpX.reshape)
  # tmpX = []
  # for j in range(test_day[i].shape[1]//9): # 最後一個當第10小時y
  #   arr = test_day[i][:,j:j+9]
  #   resh = np.reshape(arr, 9*len(arr))
  #   # print(len(np.concatenate((np.array([[1]]),resh),axis=1)))     
  #   tmpX.append(np.concatenate((np.array([[1]]),resh),axis=1)) # 加1到各個向量開頭        
  #   # print(test_day[i][9,j+9])
  # arrayX = tmpX[0]
  # for i in tmpX:
  #   arrayX = np.concatenate((arrayX,i),axis=0)
  # arrayX = np.delete(arrayX,1,axis = 0) # delete 第0列
  # batchX.append(arrayX)
print(np.array(final).shape)
final = np.array(final,dtype = np.float)
final = np.concatenate((np.ones((final.shape[0],1)),final), axis=1)
print(final.shape)

print('-*--------')
sol = final.dot(w.T)
# print(final[0])
    
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
