import numpy as np
import csv

def p(m):
  print(m)
  exit()

def readcsv(filedir, part):
  if part == "x":
    text = open(filedir, "r")
    rows = csv.reader(text, delimiter= ",")
    x = list()
    i = 0
    for row in rows:
      x.append(np.array(row))
    x = x[1:]
    x = np.array(x, float)
    # x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
    text.close()
    return x
  elif part == "y":
    text = open(filedir, "r")
    rows = csv.reader(text, delimiter= ",")
    y = list()
    i = 0
    for row in rows:
      y.append(np.array(row))
    y = y[0:]
    y = np.array(y, dtype = float).reshape(-1)
    text.close()
    return y

def readtest(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  t = list()
  for row in rows:
    t.append(np.array(row))
  t = t[1:]
  t = np.array(t, dtype = float)
  # t = np.concatenate((np.ones((t.shape[0],1)),t), axis=1)
  text.close()
  return t
def output(filedir,sol):
  out = []
  for i in range(len(sol)):
    out.append([str(i+1)])
    out[i].append(int(sol[i]))

  text = open(filedir, "w+")
  s = csv.writer(text,delimiter=',',lineterminator='\n')
  s.writerow(["id","label"])
  for i in range(len(out)):
      s.writerow(out[i]) 
  text.close()

def selftest(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  x = list()
  i = 0
  for row in rows:
    x.append(np.array(row))
  x = x[1:]
  x = np.array(x, dtype = float)
  # x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
  text.close()
  return x

def changefeature(x):
  tmp = np.expand_dims(x[:,0], axis=1) # col 0
  col = np.zeros((x.shape[0],1))
  for i in range(x.shape[1]):
    if i >= 1 and i <= 9: # col 1-9
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
    elif i == 10: # col 10
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
    elif i == 27: # col 11
      col = np.zeros((x.shape[0],1))
      col = col + np.expand_dims(x[:,i], axis=1) * 2
    elif i == 28:
      col = col + np.expand_dims(x[:,i], axis=1) * 12
    elif i == 29:
      col = col + np.expand_dims(x[:,i], axis=1) * 5
    elif i == 30:
      col = col + np.expand_dims(x[:,i], axis=1) * 7
    elif i == 31:
      col = col + np.expand_dims(x[:,i], axis=1) * 4
    elif i == 32:
      col = col + np.expand_dims(x[:,i], axis=1) * 14
    elif i == 33:
      col = col + np.expand_dims(x[:,i], axis=1) * 3
    elif i == 34:
      col = col + np.expand_dims(x[:,i], axis=1) * 8
    elif i == 35:
      col = col + np.expand_dims(x[:,i], axis=1) * 11
    elif i == 36:
      col = col + np.expand_dims(x[:,i], axis=1) * 15
    elif i == 37:
      col = col + np.expand_dims(x[:,i], axis=1) * 13
    elif i == 38:
      col = col + np.expand_dims(x[:,i], axis=1) * 16
    elif i == 39:
      col = col + np.expand_dims(x[:,i], axis=1) * 1
    elif i == 40:
      col = col + np.expand_dims(x[:,i], axis=1) * 6
    elif i == 41:
      col = col + np.expand_dims(x[:,i], axis=1) * 9
    elif i == 42:
      col = col + np.expand_dims(x[:,i], axis=1) * 10
      tmp = np.concatenate((tmp,col), axis=1)
    elif i == 43: # col 12
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
    elif i == 44: # col 13
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
    elif i == 48: # col 14
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
    elif i == 49: # col 15
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
    elif (i >= 50 and i <= 53) or (i >= 55 and i <= 64): # col 16-19 20-29
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
    elif i == 66: # relationship # col 30
      col = np.zeros((x.shape[0],1))
      col = col + np.expand_dims(x[:,i], axis=1) * 8 # Own-child
    elif i == 67:
      col = col + np.expand_dims(x[:,i], axis=1) * 16 # Not-in-family
    elif i == 68:
      col = col + np.expand_dims(x[:,i], axis=1) * 2 # Husband
    elif i == 69:
      col = col + np.expand_dims(x[:,i], axis=1) * 4 # Wife
      tmp = np.concatenate((tmp,col), axis=1)
    elif i == 71: # race # col 31
      col = np.zeros((x.shape[0],1))
      col = col + np.expand_dims(x[:,i], axis=1) * 8 # Other
    elif i == 72:
      col = col + np.expand_dims(x[:,i], axis=1) * 32 # White
    elif i == 73:
      col = col + np.expand_dims(x[:,i], axis=1) * 2 # Black
    elif i == 74:
      col = col + np.expand_dims(x[:,i], axis=1) * 1 # Asian-Pac-Islander
    elif i == 75:
      col = col + np.expand_dims(x[:,i], axis=1) * 16 # Amer-Indian-Eskimo
      tmp = np.concatenate((tmp,col), axis=1)
    elif i == 76: # sex # col 32
      col = np.zeros((x.shape[0],1))
      col = col + np.expand_dims(x[:,i], axis=1) * 3 # Female
    elif i == 77:
      col = col + np.expand_dims(x[:,i], axis=1) * 1 # Male
      tmp = np.concatenate((tmp,col), axis=1)
    elif i >= 78 and i <= 80: # col 33-35
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
    elif i >= 81:
      tmp = np.concatenate((tmp,np.expand_dims(x[:,i], axis=1)), axis=1)
  return tmp
