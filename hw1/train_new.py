import sys
import pandas as pd
import datetime
import numpy as np
from numpy import *
import csv

l = 1000

#print(df[0:0])

def placewell(array, subdata):
  #print(subdata)
  for elm, value in subdata.items():
    if elm == 'AMB_TEMP':
      array[0][1] = float(value)#*10/l
    if elm == 'CH4':
      array[0][2] = float(value)#*235/l
    if elm == 'CO':
      array[0][3] = float(value)#*430/l
    if elm == 'NMHC':
      array[0][4] = float(value)#*1000/l
    if elm == 'NO':
      array[0][5] = float(value)#*57/l
    if elm == 'NO2':
      array[0][6] = float(value)#*26/l
    if elm == 'NOx':
      array[0][7] = float(value)#*20/l
    if elm == 'O3':
      array[0][8] = float(value)#*9/l
    if elm == 'PM10':
      array[0][9] = float(value)#*7/l
    if elm == 'PM2.5':
      array[0][10] = float(value)#*10/l
    if elm == 'RAINFALL':
      if value == 'NR':
        array[0][11] = 0
      else:
        array[0][11] = float(value)#*133/l
    if elm == 'RH':
      array[0][12] = float(value)#*3.2/l
    if elm == 'SO2':
      array[0][13] = float(value)#*114/l
    if elm == 'THC':
      array[0][14] = float(value)#*210/l
    if elm == 'WD_HR':
      array[0][15] = float(value)#*1.47/l
    if elm == 'WIND_DIREC':
      array[0][16] = float(value)#*1.2/l
    if elm == 'WIND_SPEED':
      array[0][17] = float(value)#*100/l
    if elm == 'WS_HR':
      array[0][18] = float(value)#*153/l
  return array

def inputdata():
  test_x = []
  text = open(sys.argv[1], "r", encoding="big5")
  row = csv.reader(text , delimiter= ",")
  n_row = 0
  day_tmp = []
  yearb = datetime.datetime(2014, 1, 1)
  for r in row:
    if n_row != 0:
      if (n_row-1) % 18 == 0:        
        for l in day_tmp:
          test_x.append(l)
        day_tmp = []
        
        daynow = datetime.datetime(int(r[0].split('/')[0]), int(r[0].split('/')[1]), int(r[0].split('/')[2]))
        days = (daynow - yearb).days
        
        for c in range(24): #create 24 empty list for hours in one day
          day_tmp.append([str(days + (c+1)/24)])
      for ele in range(3,27): # the data from col 3 to 27 map with 0 to 23 clock
        if r[ele] != "NR":
          day_tmp[ele - 3].append(r[ele]) #because of ele is begin from 3 so the bias must be adjust with 3
        else:
          day_tmp[ele - 3].append(0)
          # for ele in range(2,11):
          #   day_tmp[ele - 2].append(r[ele])
        # print(day_tmp)
        # print(r)
    n_row = n_row + 1
  # for i in test_x:
  #   print(i)
  test_x = np.array(test_x,dtype = np.float64)
  print(test_x)
  return test_x

def testdata():
  rawdata = {}
  df = pd.read_csv('./test.csv',encoding='big5')
  df = pd.DataFrame(data = df)
  yearb = datetime.datetime(2014, 1, 1)
  
  
  print(df)
  for index, row in df.iterrows():
    daynow = datetime.datetime(int(row['日期'].split('/')[0]), int(row['日期'].split('/')[1]), int(row['日期'].split('/')[2]))
    days = (daynow - yearb).days
    for hr in range(0, 24):
      time = days + (hr+1)/24
      if rawdata.get(time) == None:
        timeline = rawdata[time] = {}
      else:
        timeline = rawdata[time]
        timeline[row['測項']] = row[str(hr)]
  
def featureNormalize(self, X):
        "Get every feature into approximately [-1, 1] range."
        featureN = len(X)
        mu = np.mean(X[0:featureN + 1], axis=1, dtype=np.float64)
        sigma = np.std(X[0:featureN + 1], axis=1, dtype=np.float64)
        X_norm = X
        for i in range(featureN):
            X_norm[i] = (X[i]-mu[i])/sigma[i]
        return X_norm, mu, sigma

def mcomputeCost(X, y, theta):
    m = size(y)
    #return (1.0 / (2.0 * m)) * sum([((np.dot(X , theta)[i]) - y[i])**2 for i in range(m)])
    return (1.0 / (2.0 * m)) * sum(power(np.dot(np.array(X,dtype = np.float) , theta) - np.array(y,dtype = np.float), 2))

def mgradientDescent(X, y, theta, alpha, num_iters):
    m = size(y)
    n = size(theta)
    print(n)
    tmp = zeros((n,1))
    X = np.array(X, dtype=np.float)
    y = np.array(y.reshape(-1), dtype=np.float)[0]
    theta = np.array(theta.reshape(-1), dtype=np.float)
    print(theta)
    X_t = X.transpose()
    s_gra = np.zeros(len(X[0]))
    for iter in range(1, num_iters + 1):
        
        # for initer in range(0, n):
        #     tmp[initer] = theta[initer] - alpha * (1.0 / m) * sum(transpose(np.dot(X, theta) - y) * X[:,initer])
        #     # print(alpha * (1.0 / m) *sum(transpose(np.dot(X, theta) - y) * X[:,initer]))
        # theta = tmp
        hypo = np.dot(X,theta)        
        loss = hypo - y        
        cost = np.sum(loss**2) / len(X)
        cost_a  = math.sqrt(cost)
        gra = np.dot(X_t,loss.T)
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        theta = theta - alpha * gra/ada
        if iter % 100 == 0 :
          print(mcomputeCost(X,y,theta))
    return theta        
    
def getfeature(dellist, X):
  tmp = np.zeros((size(X[:,0]), 0), dtype=np.float)  
  for ele in dellist:
    tmp = np.concatenate((tmp,X[:,ele]),axis=1)
  return tmp


if __name__ == '__main__':
    data = np.matrix(genfromtxt('aba67', dtype=float, delimiter=','))
    mX = mdata = np.matrix(inputdata())
    chosce = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    mX = getfeature(chosce,mdata[:,:])
    print(mX.shape)
    # print(mX)
    mX = np.delete(mX, mX.shape[0]-1 ,0)
    # print(mX)
    featurenum = size(mX[0])+1
    my = mdata[:, 10]#*10/l
    # print(my)
    my = np.delete(my, 0, 0)
    # print(my)
    
    mm = size(my)

    moneX = np.concatenate((ones((mm,1)), mX), axis = 1)
    mtheta = zeros((featurenum,1), dtype=np.float64)
    # T = moneX.T*moneX

    print(mcomputeCost(moneX, my, mtheta))

    iterations = 10000
    alpha = 10
    if len(sys.argv) >=3:
      alpha = int(sys.argv[2])
    
    
    

    #theta = gradientDescent(oneX, y, theta, alpha, iterations)
    mtheta = mgradientDescent(moneX, my, mtheta, alpha, iterations)
    
    np.save('model.npy',mtheta)
    w = np.load('model.npy')
    print(mtheta)
    
    test_x = []
    n_row = 0
    text = open('./test.csv' ,"r")
    row = csv.reader(text , delimiter= ",")
    # print(text)
    '''for r in row:
        if n_row %18 == 0:
            test_x.append([])
            for i in range(2,11):
                test_x[n_row//18].append(float(r[i]) )
        else :
            for i in range(2,11):
                if r[i] !="NR":
                    test_x[n_row//18].append(float(r[i]))
                else:
                    test_x[n_row//18].append(0)
        n_row = n_row+1'''
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
    
    test_x = getfeature(chosce,test_x)
    
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

    filename = "./predict.csv"
    text = open(filename, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","value"])
    for i in range(len(ans)):
        s.writerow(ans[i]) 
    text.close()


