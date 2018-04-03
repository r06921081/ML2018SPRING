import numpy as np
import tools as tl
def p(m):
  print(m)
  exit()

def sigmoid(z):
  # return 1 / (1 + np.exp(-z))
  return np.clip(1 / (1 + np.exp(-z)), 1e-8, 1-(1e-8))
  
def Pwb(x, w):
  return sigmoid(x.dot(w))

def gd(x, y, u_0, u_1, sigma_0, sigma_1):
  data_num = x.shape[0]
  type_0 = 0
  type_1 = 0
  for i, typ in zip(range(data_num), y):
    if typ == 0:
      u_0 += x[i]
      type_0 += 1
    else:
      u_1 += x[i]
      type_1 += 1
  u_0 = u_0 / type_0
  # u_0 = np.matrix(u_0).transpose() 
  u_1 = u_1 / type_1
  # u_1 = np.matrix(u_1).transpose()
  for i, data, typ in zip(range(data_num), x, y):
    x_mat = np.matrix(data)
    if typ == 0:
      diff = x_mat - np.matrix(u_0)
      # print(x_mat)
      # print(np.matrix(u_0))
      sigma_0 += diff.T.dot(diff)
      # p(sigma_0)
    else:
      diff = x_mat - np.matrix(u_1)
      # print(x_mat)
      # print(np.matrix(u_1))
      sigma_1 += diff.T.dot(diff)   
      # p(sigma_1)
  
  return u_0/data_num, u_1/data_num, sigma_0, sigma_1, type_0, type_1

def Generative(x,y,lr,epoch):
  batch = 1
  Xtran = np.transpose(x)
  s_grad = np.zeros(x.shape[1])
  u0 = np.matrix(np.zeros(x.shape[1]))
  u1 = np.matrix(np.zeros(x.shape[1]))
  sigma0 = np.identity(x.shape[1]) * 0.001#u0.T.dot(u0)#
  sigma1 = np.identity(x.shape[1]) * 0.001#u0.T.dot(u0)#
  D = x.shape[1]
  c1 = (2*np.pi)**(D/2)
  for i in range(1):

    num_bat = i%batch
    batstr = len(x)//batch*num_bat
    batend = batstr + len(x)//batch
    batx = x[batstr:batend]
    baty = y[batstr:batend]

    u0, u1, sigma0, sigma1, type0, type1 = gd(batx, baty, u0, u1, sigma0, sigma1)
    # p(sigma1[1,:])
    if i % 100 == 0:
      sigma_avg = sigma0 * (type0/type0+type1) + sigma1 * (type1/type0+type1)
      # p(np.linalg.det(sigma_avg))
      
      sigma_inv = np.linalg.inv(sigma_avg)
      det = np.linalg.det(sigma_avg)
      # p(det)
      c = 1/(c1 *(det**0.5))
      # print(x)
      posb = 1
      for xn, yn in zip(batx, baty):
        if yn == 0:
          v = xn - u0
        else:
          v = xn - u1
        posb *= c * np.exp((-0.5) * v.dot(sigma_inv.dot(v.T)))
      v0 = batx - u0
      v1 = batx - u1
      f0 = (-1/2) * np.sum(np.multiply(np.dot(v0,sigma_inv),v0),axis = 1)
      f1 = (-1/2) * np.sum(np.multiply(np.dot(v1,sigma_inv),v1),axis = 1)
      # print(f0.shape)
      # print(y.shape)
      # p(np.multiply(y.reshape(x.shape[0],1),f0))
      # p(y.shape)
      multresult = np.dot(baty,f0) + np.dot((1-baty),f1)
      # p(c)
      loss = np.log(c) + multresult
      # p(loss)
      '''---------'''
      w = np.dot((u0 - u1), sigma_inv)
      b = (-0.5) * np.dot(np.dot(u0, sigma_inv), u0.T) + (0.5) * np.dot(np.dot(u1, sigma_inv), u1.T) + np.log(float(type0)/type1)
      z = np.dot(w, batx.T) + b
      faild = sigmoid(z)
      faild = faild.T
      for j in range(len(faild)):
        if faild[j] < 0.5:
          faild[j] = 1
        else:
          faild[j] = 0
      err = vaild(faild.T,baty)

      print(i,'-cost:',loss,'-error',err)
  return w.ravel(), b 

def regression(x, y, w, b, lam, lr = 0.1, epoch = 10000):
  batch = 180
  Xtran = np.transpose(x)
  s_gradw = np.zeros(x.shape[1])
  s_gradb = 0
  mincost = lasterror = y.shape[0]
  for i in range(epoch):
    num_bat = i%batch
    batstr = len(x)//batch*num_bat
    batend = batstr + len(x)//batch
    batx = x[batstr:batend]
    baty = y[batstr:batend]

    gradient_w, gradient_b = logist_gradient(batx, baty, w, b)
    # s_gradw += np.power(gradient_w, 2)
    # s_gradb += np.power(gradient_b, 2)
    
    # adagw = np.sqrt(s_gradw) + np.ones(np.size(s_gradw)) * 0.001
    # adagb = np.sqrt(s_gradb) + np.ones(np.size(s_gradb)) * 0.001
    # print(lam / batx.shape[0] * 2)
    w = w - lr * gradient_w - (lam / batx.shape[0]) * np.abs(w) #/adagw
    b = b - lr * gradient_b #/adagb
    if i % 100 == 0:
      loss = Loss(sigmoid(np.dot(batx, w) + b), baty, "Crossentropy")
      

      faild = sigmoid(x.dot(w) +b)
      faild, hmax, hmin = tl.rescaling(faild)
      for j in range(len(faild)):
        if faild[j] < 0.5:
          faild[j] = 0
        else:
          faild[j] = 1
      err = vaild(faild,y)
      print(str(i) + '-Crossentropy:' + str(loss),'error:',err)
      


  return w, b

def logist_gradient(x, y, w, b):
    '''
    x: [1, x1, x2, ...]
    w: [bias, w1, w2, ...]
    '''
    num_data = x.shape[0]
    num_fea = x.shape[1]
    gradient_w = np.zeros(num_fea)

    z = np.dot(x, w) + b
    hypothesis = sigmoid(z)
    
    loss = y - hypothesis
    w_grad = np.mean(-1 * x * loss.reshape((num_data,1)), axis = 0)
    b_grad = np.mean(-1 * loss)
    # gradient_w = np.dot(np.transpose(x), loss) / num_data

    return w_grad, b_grad

def pocket(x, y, num):
  pkt_num = num
  px = []
  py = []
  pkt_size = len(x)//pkt_num
  for i in range(pkt_num):
    begin = i * pkt_size
    px.append(x[begin:begin + pkt_size])
    py.append(y[begin:begin + pkt_size])
  return px, py

def regression_p(x, y, w, earlystop, varrate, lr = 0.1, epoch = 10000):
  '''
  parameter setting
  '''
  batch = 180 # bath size
  Xtran = np.transpose(x)
  s_grad = np.zeros(x.shape[1])
  chance = 20
  notmove = 0
  mincost = lasterr = x.shape[0]
  minw = [[1000000,1]]*chance # init earlystop conparer
  pkt_num = 31
  if varrate != 0:
    cut = int(varrate * x.shape[0])
    x,y = tl.shuffle(x,y)
    varx = x[:cut]
    vary = y[:cut]
    x = x[cut:]
    y = y[cut:]
  else:
    varx = x
    vary = y
  '''
  select each peace data with hyposethy
  '''
  pktx, pkty = pocket(x, y, pkt_num)
  w_arr = [w] * pkt_num
  #3 6 7 9 11 12 15 16 17 19 20 21 22 28 29 30
  largex = pktx[0]
  largey = pkty[0]
  for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]:#6, 7, 9, 11, 12, 15, 16, 17, 19, 20, 21, 22, 28, 29, 30
  # largex = pktx[3]
  # largey = pkty[3]
  # for i in [6,7,9,11,12,15,16,17,19,20,21,22,28,29,30]:#6, 7, 9, 11, 12, 15, 16, 17, 19, 20, 21, 22, 28, 29, 30
    largex = np.concatenate((largex,pktx[i]),axis = 0)
    largey = np.concatenate((largey,pkty[i]),axis = 0)
  '''final I found it's no different with each peace'''
  for i in range(epoch):
    # for j in range(pkt_num):
    #   num_bat = i%batch
    #   batstr = len(pktx[j])//batch*num_bat
    #   batend = batstr + len(pktx[j])//batch
    #   batx = pktx[j][batstr:batend]
    #   baty = pkty[j][batstr:batend]
    #   h = batx.dot(w_arr[j].T)
  
    #   loss = h - baty  

    #   cost = np.sum(loss.T*loss) / len(batx)
    #   costval = np.power(cost, 0.5)

    #   if i % 100 == 0:
    #     print(str(i) + '-cost:' + str(costval) + '-alpha:' + str(lr))
    #     # if lr < 50 :
    #     #   lr = ((epoch - i) / epoch) * 10240
    #     # else:
    #     #   lr = lr*0.5

    #   grad = np.dot(batx.transpose(),loss.T)
    #   s_grad += np.power(grad,2)
      
    #   adag = np.sqrt(s_grad) + np.ones(np.size(s_grad)) * 0.01

    #   w_arr[j] = w_arr[j] - lr * grad/adag
  
    num_bat = i%batch
    batstr = len(largex)//batch*num_bat
    batend = batstr + len(largey)//batch
    batx = largex[batstr:batend]
    baty = largey[batstr:batend]
    h = sigmoid(batx.dot(w.T))

    loss = h - baty

    # cost = np.sum(loss.T*loss) / len(batx)
    # costval = np.power(cost, 0.5)
    costval = Loss(h, baty, "Logist")

    grad = np.dot(batx.transpose(),loss.T)
    s_grad += np.power(grad,2)
    
    adag = np.sqrt(s_grad) + np.ones(np.size(s_grad)) * 0.01

    w = w - lr * grad/adag
    if i % 100 == 0:
      # ----------------cal the y^------------------------
      faild = sigmoid(varx.dot(w))
      faild, hmax, hmin = tl.rescaling(faild)
      for j in range(len(faild)):
        if faild[j] < 0.5:
          faild[j] = 0
        else:
          faild[j] = 1
      err = vaild(faild,vary)
      # print(str(i) + 'error:' + str(err) + '-cost:' + str(costval) + '-lr:' + str(lr))   
      if err == lasterr:
        notmove += 1 
      if notmove > 10:
        lr = lr * 1.1
        notmove = notmove -1
      elif lr < 1 and ((epoch - i) / epoch) > 0.3:
        lr = ((epoch - i) / epoch) * 100
      elif lr < 0.001:
        lr = 0.05
      elif ((epoch - i) / epoch) <= 0.3:
        lr = lr * 0.95
      else:
        lr = lr - 0.1
      lasterr = err
      '''
      early stop
      '''      
      # ----------------store the min---------------------
      if err <= mincost:
        mincost = err
        mw = w
        print(str(i) + 'error:' + str(err) + '-cost:' + str(costval) + '-lr:' + str(lr))   
      # ----------------check to break--------------------
      if earlystop == 1:
        j = 0
        for _ in range(chance - 1):
          if minw[(i%chance - j-1)%chance][0] > err:
            break
          j += 1
        if j == chance - 1:
          minpos = -1
          for k in range(chance):
            if minw[k][0] < mincost:
              minpos = k
              mincost = minw[k][0]
              print(minpos,mincost)
          if minpos == -1:
            return mw,baty
          else:
            return minw[minpos][1],baty
        minw[(i%(chance*100)//100 + 1)%chance] = [err, w] # all big then now w w_queue keep going    
      

  if err > mincost:
    return mw,baty
  return w,baty

def vaild(varesult,vay):
  # print('error:',np.sum(np.absolute(vay - varesult)))
  return np.sum(np.absolute(vay - varesult))

def Loss(hy, y, ltype):
  loss = 0
  if ltype == "Logist":
    return np.sum((np.multiply(y, hy) + np.multiply(1 - y, 1 - hy)))/len(y)
  elif ltype == "MSE":
    return 0.5*(np.sum(hy-y)/y.shape[0])**2
  elif ltype == "Crossentropy":
    return -(np.dot(y, np.log(hy)) + np.dot((1 - y), np.log(1 - hy)))



