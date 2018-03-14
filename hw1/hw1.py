import numpy as np
import csv
import sys

def getfeature(dellist, X):
  tmp = np.zeros((np.size(X[:,0]), 0), dtype=np.float)  
  for ele in dellist:
    tmp = np.concatenate((tmp,X[:,ele]),axis=1)
  return tmp

if __name__ == '__main__':
  w = np.load('model.npy')
    
  test_x = []
  n_row = 0
  text = open(sys.argv[1] ,"r")
  row = csv.reader(text , delimiter= ",")
# print(text)

  n_row = 0
  day_tmp = []
  for r in row:
    if n_row % 18 == 0:
      for l in day_tmp:
        test_x.append(l)
        day_tmp = []
      for c in range(9):
        day_tmp.append([str(n_row//18 + (c+1)/24)])
    for ele in range(2,11):
      if r[ele] != "NR":
        day_tmp[ele - 2].append(r[ele])
      else:
        day_tmp[ele - 2].append(0)
    # for ele in range(2,11):
     #   day_tmp[ele - 2].append(r[ele])
      # print(day_tmp)
      # print(r)
    n_row += 1
  for l in day_tmp:
    test_x.append(l)
    

  text.close()
  test_x = np.matrix(test_x)
    
  test_x = getfeature([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],test_x)
    
  test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
  w = np.matrix(w)
  print(test_x[0])
  print(w)
    
  print('-*--------')
  ans = []
  sol = np.dot(np.array(test_x[0],dtype=np.float),w.T)[0,0]#/10*l
  for i in range(1,len(test_x)):      
    if i % 9 == 0:
      ans.append(["id_"+str(i//9-1)])
      ans[len(ans)-1].append(sol)
      sol = np.dot(np.array(test_x[i+1],dtype=np.float),w.T)[0,0]#/10*l
    sol = sol * 0.4 + 0.6 * np.dot(np.array(test_x[i],dtype=np.float),w.T)[0,0]#/10*l
  ans.append(["id_"+str(int(ans[len(ans)-1][0].split('_')[1]) + 1)])
  ans[len(ans)-1].append(sol)

  filename = sys.argv[2]
  text = open(filename, "w+")
  s = csv.writer(text,delimiter=',',lineterminator='\n')
  s.writerow(["id","value"])
  for i in range(len(ans)):
    s.writerow(ans[i]) 
  text.close()