import sys
import pandas as pd
import numpy as np
import csv
import threading
from queue import Queue
import matplotlib.pyplot as plt

lr_decentrate = 1
epoch = 10000
lr = 0.0000022
lam = 0.01
X2_on = 0
chosce = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] # 選擇要使用的feature 0,1,3,5,7,8,9,10,12
def appendx2(X, x2select, lamd):
  X = np.concatenate((X,np.power(X[x2select,:],2)),axis=0)
  return X

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
          day_tmp = np.matrix(day_tmp,np.float)
          final = np.concatenate((final,day_tmp),axis=1)
          # print(day_tmp.shape)
        else:
          final = np.array(day_tmp)
        day_tmp = []
        if (n_row-1) % 360 == 0 and n_row != 1:            
          month.append(final)
          final = []
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

  return month

def computeCost(X, y, theta):
  m = np.size(y)
  return np.power(np.sum(np.power(X.dot(theta) - y, 2))/m,0.5)/2

def thread_job(theta,lr,m,loss,X, q):
  tmp = theta - lr * (1.0 / m) * np.sum(loss.dot(X))
  q.put(tmp)

def gradientDescent(X, y, theta, lr, epoch):
    m = np.size(y)
    n = np.size(theta)
    # print(n)
    theta = np.array(theta)
    tmp = np.zeros((n,1))
    
    X = np.array(X, dtype=np.float)
    y = np.array(y, dtype=np.float)
    # print(theta)
    theta = np.array(theta.reshape(-1), dtype=np.float)
    
    Xtran = X.transpose()
    # print(Xtran.shape)
    s_grad = np.zeros(len(X[0]))
    costval = 0
    costlist = []
    for iter in range(epoch):
        h = X.dot(theta)  
        
        loss = h - y

        regu = np.copy(theta)
        regu[0] = 0        
        theta = theta - lr * (1.0 / m) * (Xtran.dot(loss) + lam * (regu**2))
        cost = computeCost(X,y,theta)
        costlist.append(cost)
        if iter % 100 == 0:
          print(str(iter)+':'+str(cost))
        
    return theta, costlist
    
def getfeature(dellist, X):
  tmp = np.ones((1, np.size(X[0,:])), dtype=np.float)  

  for ele in dellist:
    tmp = np.concatenate((tmp,X[ele,:]),axis=0)
  return tmp[1:]

def valid(X,y,theta,batchsize):
  loss = X.dot(theta.T) - y
  MSE = np.sqrt(loss * loss.T)
  return MSE

if __name__ == '__main__':
    if len(sys.argv) >=3: # 外部lr參數傳入
      lr = float(sys.argv[2])
    mX = mdata = inputdata()
    
    
    
    if X2_on == 2:
      for i in range(len(chosce)): # 把二次項的選擇加入chosce list 中給他選
        chosce.append(mdata[0].shape[0] + i)
    elif X2_on == 1:
      chosce.append(18)
    for i in range(len(mX)): # 把各個月份的feature 選好
      if X2_on == 2:
        for x2 in chosce: # 把各個二次項加上去
          mX[i] = appendx2(mX[i],x2,lam)
      elif X2_on == 1:
        mX[i] = appendx2(mX[i],9,lam)
      
      mX[i] = getfeature(chosce,mX[i])
    print(mX[0].shape)
    print('-----')
    
    batchX = []
    batchy = []
    print(mX[0].shape[1])
    print('+++')

    '''
    把每個月的資料都處理好 把對應的前九天放入一個向量，並把第十天PM2.5的直放到y當作答案
    因為前九天的y值沒有對應-9~-1的天數所以會損失9筆資料所以一個月有20*24-9 = 471筆資料
    '''
    for i in range(len(mX)): 
      tmpX = []
      tmpy = []
      for j in range(mX[i].shape[1]-9): # 最後一個當第10小時y
        arr = mX[i][:,j:j+9]
        resh = np.reshape(arr, 9*len(arr))
        tmpX.append(np.concatenate((np.array([[1]]),resh),axis=1)) # 加1到各個向量開頭        
        tmpy.append(mX[i][chosce.index(9),j+9])
      
      arrayX = tmpX[0]
      for i in tmpX:
        arrayX = np.concatenate((arrayX,i),axis=0)
      arrayX = np.delete(arrayX,1,axis = 0) # delete 第0列
      batchX.append(arrayX)
      batchy.append(tmpy)
    
    print('+++')
    featurenum = np.size(batchX[0][0]) # 建立theta向量
    theta = np.zeros((featurenum,1), dtype=np.float)
    m = np.size(batchy[0]) # 取得每個月資料長度

    temp_Xb = []
    temp_yb = []


    choseMon = [0,1,2,3,4,5,6,7,8,9,10,11] # 選擇使用的月份資料 0 1 4 5 6 7 9 10 
    for mon in choseMon: # 把所有月份資料串起來
      if len(temp_Xb) == 0:
        temp_Xb = batchX[mon]
        temp_yb = batchy[mon]
      else:
        temp_Xb = np.concatenate((temp_Xb,batchX[mon]),axis=0)
        temp_yb = np.concatenate((temp_yb,batchy[mon]),axis=0)
    batchX = [temp_Xb]
    batchy = [temp_yb]
    batchnum = len(batchX)

    b_size = batchX[0].shape[0] # 取得每筆資料長度
    loss = 0
    avgtheta = [] # 平均theta的error
    pocket = [] # 候選的theta們
    batchtheta = []
    plotcost = []

    for i in range(batchnum): # 建立batch數量的theta
      batchtheta.append(theta)

    print('-------------train----------------------')
    for i in range(batchnum): # 訓練每個batch
      tmptheta, plotcost = gradientDescent(batchX[i], batchy[i], batchtheta[i], lr, epoch)
      pocket.append(tmptheta)
      avgtheta.append(0)
    for i in range(batchnum): # n-fold 算每個batch輪流當valid平均的cost
      validX = batchX[i]      
      validy = batchy[i]
      for j in range(len(batchX)): # 把cost累積起來
          avgtheta[j] += valid(validX,validy,pocket[j],b_size)
          print(valid(validX,validy,pocket[j],b_size))
    print('-------------costs-----------------------')
    print('ththththththththth')
    print(avgtheta)
    print('ththththththththth')
    
    minin = avgtheta[0]
    selecttheta = 0
    for i in range(len(batchX)): # 取出最小的 cost
      if avgtheta[i] < minin:
        selecttheta = i
    print(avgtheta[selecttheta]) # 最小的 cost
    print('*-----------------')
    
    
    np.save('model.npy',pocket[0])
    np.save('chosce.npy',chosce)