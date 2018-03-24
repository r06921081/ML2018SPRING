import numpy as np
def p(m):
  print(m)
  exit()
def sigmoid(z):
  # return 1 / (1 + np.exp(-z))
  return np.clip(1 / (1 + np.exp(-z)), 1e-8, 1-(1e-8))
  

def Pwb(x, w):
  return sigmoid(x.dot(w))

def regression(x, y, w, lr = 0.1, epoch = 10000):
  Xtran = np.transpose(x)
  s_grad = np.zeros(x.shape[1])
  for i in range(epoch):
    
    # pwb = sigmoid(x.dsot(w))

    
    # if i % 100 == 0:
    #   loss = Loss(x, y, w, "MSE")
    #   print(loss)

    # grad = gradient(x,y,w)
    
    # w = w - lr * grad
    
    h = x.dot(w.T)
  
    loss = h - y  

    cost = np.sum(loss.T*loss) / len(x)
    costval = np.power(cost, 0.5)

    if i % 100 == 0:
      print(str(i) + '-cost:' + str(costval) + '-alpha:' + str(lr))
      # if lr < 1 :
      #   lr = ((1000000 - i) / 1000000) * 10240
      # else:
      #   lr = lr*0.8

    grad = np.dot(Xtran,loss.T)
    s_grad += np.power(grad,2)
    
    adag = np.sqrt(s_grad) + np.ones(np.size(s_grad)) * 0.01

    w = w - lr * grad/adag
  return w 

def Loss(x, y, w, ltype):
  loss = 0
  if ltype == "Logist":
    pwb = Pwb(x, w)
    return np.log(np.sum((np.multiply(y, pwb) + np.multiply(1 - y, 1 - pwb))))
  elif ltype == "MSE":
    return 0.5*(np.sum(x.dot(w)-y)/y.shape[0])**2

def gradient(x, y, w):
    '''
    x: [1, x1, x2, ...]
    w: [bias, w1, w2, ...]
    '''
    num_data = x.shape[0]
    num_fea = x.shape[1]
    gradient_w = np.zeros(num_fea)

    z = np.dot(x, w)
    hypothesis = sigmoid(z)

    loss = hypothesis - y    
    
    gradient_w = np.dot(np.transpose(x), loss) / num_data

    return gradient_w