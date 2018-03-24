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
    x = np.array(x, dtype = float)
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
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
  t = np.concatenate((np.ones((t.shape[0],1)),t), axis=1)
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
  x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
  text.close()
  return x