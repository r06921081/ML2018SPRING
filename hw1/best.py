import sys
import pandas as pd
import numpy as np
from numpy import *
import csv

l = 1000

rate = 0.2

#print(df[0:0])

def scale(value, No):
  #print(subdata)
  
  #for elm, value in subdata.items():
    if No == 0:#'AMB_TEMP':
      return float(value) * 10 / l#array[0][1] = float(value)#*10/l
    if No == 1:#'CH4':
      return float(value) * 235 / l#array[0][2] = float(value)#*235/l
    if No == 2:#'CO':
      return float(value) * 430 / l#array[0][3] = float(value)#*430/l
    if No == 3:#'NMHC':
      return float(value) * 1000 / l#array[0][4] = float(value)#*1000/l
    if No == 4:#'NO':
      return float(value) * 57 / l#array[0][5] = float(value)#*57/l
    if No == 5:#'NO2':
      return float(value) * 26 / l#array[0][6] = float(value)#*26/l
    if No == 6:#'NOx':
      return float(value) * 20 / l#array[0][7] = float(value)#*20/l
    if No == 7:#'O3':
      return float(value) * 9 / l#array[0][8] = float(value)#*9/l
    if No == 8:#'PM10':
      return float(value) * 7 / l#array[0][9] = float(value)#*7/l
    if No == 9:#'PM2.5':
      return float(value) * 10 / l#array[0][10] = float(value)#*10/l
    if No == 10:#'RAINFALL':
      if value == 'NR':
        return 0
      else:
        return float(value) * 133 / l#array[0][11] = float(value)#*133/l
    if No == 11:#'RH':
      return float(value) * 3.2 / l#array[0][12] = float(value)#*3.2/l
    if No == 12:#'SO2':
      return float(value) * 114 / l#array[0][13] = float(value)#*114/l
    if No == 13:#'THC':
      return float(value) * 210 / l#array[0][14] = float(value)#*210/l
    if No == 14:#'WD_HR':
      return float(value) * 1.47 / l#array[0][15] = float(value)#*1.47/l
    if No == 15:#'WIND_DIREC':
      return float(value) * 1.2 / l#array[0][16] = float(value)#*1.2/l
    if No == 16:#'WIND_SPEED':
      return float(value) * 100 / l#array[0][17] = float(value)#*100/l
    if No == 17:#'WS_HR':
      return float(value) * 153 / l#array[0][18] = float(value)#*153/l
    return 0

def inputdata():
  test_x = []
  text = open(sys.argv[1], "r", encoding="big5")
  row = csv.reader(text , delimiter= ",")
  n_row = 0
  day_tmp = []
  ele_temp = []
  final = []
  month = []
  for r in row:
    if n_row != 0:
      if (n_row-1) % 18 == 0:                
        if final != []:
          day_tmp = np.matrix(day_tmp,np.float64)
          final = np.concatenate((final,day_tmp),axis=1)
          # print(day_tmp.shape)
        else:
          final = np.array(day_tmp)
        day_tmp = []
        if (n_row-1) % 360 == 0 and n_row != 1:            
          month.append(final)
          final = []
        # day_tmp[ele - 3].append(scale(r[ele],(n_row-1) % 18))
          # for ele in range(2,11):
          #   day_tmp[ele - 2].append(r[ele])
        # print(day_tmp)
        # print(r)
      ele_temp = []
      for ele in range(3,27): # the data from col 3 to 27 map with 0 to 23 clock
        if r[ele] != "NR":
          ele_temp.append(float(r[ele])) #because of ele is begin from 3 so the bias must be adjust with 3
        else:
          ele_temp.append(0)
      day_tmp.append(ele_temp)
    n_row = n_row + 1
  final = np.concatenate((final,day_tmp),axis=1)
  month.append(final)
  # for i in test_x:
  #   print(i)
  return month

def mcomputeCost(X, y, theta):
    m = size(y)
    #return (1.0 / (2.0 * m)) * sum([((np.dot(X , theta)[i]) - y[i])**2 for i in range(m)])
    return (1.0 / (2.0 * m)) * sum(power(np.dot(np.array(X,dtype = np.float) , theta) - np.array(y,dtype = np.float), 2))

def gradientDescent(X, y, theta, alpha, num_iters,i):
    m = size(y)
    n = size(theta)
    # print(n)
    theta = np.array(theta)
    tmp = zeros((n,1))
    
    X = np.array(X, dtype=np.float)
    y = np.array(y, dtype=np.float)
    # print(theta)
    theta = np.array(theta.reshape(-1), dtype=np.float)
    
    Xtran = X.transpose()
    # print(Xtran.shape)
    s_grad = np.zeros(len(X[0]))
    costval = 0
    for iter in range(1, num_iters + 1):
        h = X.dot(theta.T)
  
        loss = h - y  
    
        cost = np.sum(loss.T*loss) / len(X)
        costval = np.power(cost, 0.5)

        if iter % 1000 == 0:
          print(str(iter) + '-cost:' + str(costval) + '-alpha:' + str(alpha))
          if alpha < 1 :
            alpha = 512
          else:
            alpha = alpha*0.8

        grad = np.dot(Xtran,loss.T)
        s_grad += np.power(grad,2)
        adag = np.sqrt(s_grad) + np.ones(np.size(s_grad)) * 0.01

        theta = theta - alpha * grad/adag

    # print(theta)
    return theta     
    
def getfeature(dellist, X):
  tmp = np.ones((1, size(X[0,:])), dtype=np.float)  

  for ele in dellist:
    tmp = np.concatenate((tmp,X[ele,:]),axis=0)
  return tmp[1:]

def valid(X,y,theta,batchsize):
  loss = X.dot(theta.T) - y
  MSE = np.sqrt(loss * loss.T)
  return MSE

def appendx2(X, x2select):
  # print(X[9,:])
  X = np.concatenate((X,np.power(X[x2select,:],2)),axis=0)
  # print(X[18,:])
  return X

def p(m):
  print(m)
  exit()
if __name__ == '__main__':
    # data = np.matrix(genfromtxt('aba67', dtype=float, delimiter=','))
    mX = mdata = inputdata()
    
    chosce = [0,1,3,5,7,8,9,10,12,18]
    
    # mX = getfeature(chosce,mdata)
    for i in range(len(mX)):
      # for ele in chosce:
      #   mX[i] = appendx2(mX[i],ele)
      mX[i] = appendx2(mX[i],9)
      mX[i] = getfeature(chosce,mX[i])
    print(mX[0].shape)
    print('-----')
    
    batchX = []
    print(mX[0].shape[1])
    print('+++')
    batchy = []
    for i in range(len(mX)):
      tmpX = []
      tmpy = []
      
      for j in range(mX[i].shape[1]-9): # 最後一個當第10小時y        
        arr = mX[i][:,j:j+9]
        resh = np.reshape(arr, 9*len(arr))
        # e(resh.shape) 
        tmpX.append(np.concatenate((np.array([[1]]),resh),axis=1)) # 加1到各個向量開頭        
        tmpy.append(mX[i][chosce.index(9),j+9])

      arrayX = tmpX[0]
      for i in tmpX:
        arrayX = np.concatenate((arrayX,i),axis=0)
      arrayX = np.delete(arrayX,1,axis = 0) # delete 第0列
      arrayX = np.matrix(arrayX)
      batchX.append(arrayX)
      batchy.append(tmpy)
    # b = mX[0][:,0:9]
    
    print('+++')

    featurenum = np.size(batchX[0][0])
    theta = zeros((featurenum,1), dtype=np.float64)
    m = np.size(batchy[0])
  
    iterations = 100000
    alpha = 4096
    if len(sys.argv) >=4:
      alpha = float(sys.argv[3])
    
    temp_Xb = []
    temp_yb = []
    batchnum = 1


    choseMon = ['0:0,1,2,3','1:0,1,2,3','4:0,1,2,3','5:0,1,2,3','6:0,1,2,3','7:0,1,2,3','9:0,1,2,3','10:0,1,2,3'] # 選擇使用的月份資料 0 1 4 5 6 7 9 10 
    silcenum = 4
    silcesize = len(batchX[0])//silcenum
    for mon in choseMon: # 把所有月份資料串起來
      month = int(mon.split(':')[0])
      for silce in mon.split(':')[1].split(','):
        if len(temp_Xb) == 0:
          temp_Xb = batchX[month][int(silce)*silcesize:(int(silce)+1)*silcesize]
          temp_yb = batchy[month][int(silce)*silcesize:(int(silce)+1)*silcesize]
        else:
          temp_Xb = np.concatenate((temp_Xb,batchX[month][int(silce)*silcesize:(int(silce)+1)*silcesize]),axis=0)
          temp_yb = np.concatenate((temp_yb,batchy[month][int(silce)*silcesize:(int(silce)+1)*silcesize]),axis=0)
    batchX = [temp_Xb]
    batchy = [temp_yb]
    batchnum = len(batchX)
      
    # batchX = []
    # batchy = []
    b_size = batchX[0].shape[0]
    loss = 0
    avgtheta = []
    pocket = []

    # print(batchy[0])
    # mtheta, loss = mgradientDescent(batchX[9], batchy[9], mtheta, alpha, iterations)
    # for i in batchX:
    #   print(i[0])
    # exit()
    batchtheta = [theta,theta,theta,theta]
    print('-------------train----------------------')    
    for i in range(len(batchX)): # train step
      pocket.append(gradientDescent(batchX[i], batchy[i], batchtheta[i], alpha, iterations,i))
      avgtheta.append(0)
    for i in range(len(batchX)):
      validX = batchX[i]      
      validy = batchy[i]
      for j in range(len(batchX)):
        # if i != j:
          avgtheta[j] += valid(validX,validy,pocket[j],b_size)
          print(valid(validX,validy,pocket[j],b_size))
    print('ththththththththth')
    # print(avgtheta)
    print('ththththththththth')
    
    minin = avgtheta[0]
    selecttheta = 0
    for i in range(len(batchX)):
      if avgtheta[i] < minin:
        selecttheta = i
    print(avgtheta[selecttheta])
    print('*-----------------')
    
    np.save('model_best.npy',pocket[0])
    np.save('chose_best.npy',chosce)
    w = np.load('model_best.npy')
    print('lllllllllllllllllll')
    print(w)
    print('llllllllllllllllllll')
    test_x = []
    n_row = 0
    text = open('./data/test.csv' ,"r")
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
          tmp_9hr = np.matrix(tmp_9hr,np.float64)
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
   
    test_day = np.matrix(test_day,np.float64)
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
      # arrc = []
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[0][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[1][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[2][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[3][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[4][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[5][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[6][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[7][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[8][:,k])[0][0])
      # for k in range(0,9):
      #   arrc.append(np.ndarray.tolist(tmpX[9][:,k])[0][0])
      # resh = np.reshape(np.array(arrc), (1,90))
      # final.append(np.reshape(np.array(resh), 90))
    print(np.array(final).shape)
    final = np.array(final,dtype = np.float64)
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
