import numpy as np
np.random.seed(1126)
import csv
import PIL
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
import matplotlib.image as mpimg
from tools import p

def readcsv(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  x = list()
  y = list()
  for i, row in enumerate(rows):
    if i != 0:
      y.append(np.array(row[0]))
      data = row[1].split(' ')
      x.append(list(map(int, data)))  
    # if i == 3:
    #   break

  #x = np.array(x, float)
  # x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
  text.close()
  return np.array(x), np.array(y)

def readtest(filedir):
  text = open(filedir, "r")
  rows = csv.reader(text, delimiter= ",")
  x = list()
  for i, row in enumerate(rows):
    if i != 0:
      data = row[1].split(' ')
      x.append(list(map(int, data)))  
    # if i == 3:
    #   break

  #x = np.array(x, float)
  # x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
  text.close()
  return np.array(x)

def savepre(result, filedir):
  data2write = [['id', 'label']]
  for i, row in enumerate(result):
    data2write.append([int(i), int(np.argmax(row))])
  text = open(filedir, "w+")
  s = csv.writer(text,delimiter=',',lineterminator='\n')
  for i in data2write:
      s.writerow(i) 
  text.close()


def picsave(x, y):
  out = []
  for i, ix in enumerate(x):
    out.append([y[i]])
    datastr = ''
    for j in ix:
      datastr += str(j) + ' '
    out[i].append(datastr.rstrip())
  # print(out)
  text = open('./ooo.csv', "w+")
  s = csv.writer(text,delimiter=',',lineterminator='\n')
  s.writerow(["label","newfeature"])
  for i in range(len(out)):
      s.writerow(out[i]) 
  text.close()

def toPic(piclist):
  return np.array(piclist).reshape(48,48)

def read_dataset(path, labeled_data = True, shuffle = False):
  '''
  # Arguments
  path : 相對位址 (relative address)
  labeled_data : 表示 x 是 data
      True : 表示是要處理 trainig set\n
      False : 表示是要處理 testinig set
  shuffle : 決定是否打亂 datas ,但 testing set 無法打亂
  # Return
  traning set : 2-D array, label, line_index (共3項)\n
  testing set : 2-D array
  '''
  print('\nReading File...')

  with open(path) as file:
    print('--Open file:',path)
    if labeled_data:
      sta = 'Traning Set'
    else:
      sta = 'Testing Set'
    print('--Stament :',sta )

    datas = []

    for ind, line in enumerate(file):

      if labeled_data:
        lable, feat = line.split(',')
      else:
        _, feat = line.split(',')

      feat = np.fromstring(feat, dtype = int, sep = ' ')
      try:
        feat = np.reshape(feat, (1, 48, 48))
      except:
        print('--Exception at line', ind)
        continue
      
      if labeled_data:
        datas.append((feat, int(lable), ind))
      else:
        datas.append(feat)
    if shuffle and labeled_data:
      np.random.shuffle(datas)
        
    if labeled_data:
      feats, lables, line_ind = zip(*datas) #這裡因為 datas 儲存了多變數
    else:
      feats = datas

    feats = np.asarray(feats)
    print('--feats shape :', np.shape(feats))

    if labeled_data:
      lables = to_categorical(np.asarray(lables, dtype = float))
      print('--lables shape :', np.shape(lables))
      return feats / 255, lables, line_ind
    else:
      return feats / 255

if __name__ == "__main__":
  # im = PIL.Image.open( "1.png" )
  # im = im.rotate( 20, PIL.Image.BILINEAR )
  # plt.imshow(im, cmap='gray', interpolation='nearest')
  # plt.show()
  # print(type(im))
  x,y = readcsv('./data/train.csv')
  newx = []
  newy = []
  text = open('./ooo.csv', "a+")
  s = csv.writer(text,delimiter=',',lineterminator='\n')
  s.writerow(["label","newfeature"])
  text.close()
  for i, iy in zip(x, y):
    text = open('./ooo.csv', "a+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    lists = []
    tmp = PIL.Image.fromarray(np.uint8(toPic(i))).transpose(PIL.Image.FLIP_LEFT_RIGHT)
    row = [iy.tolist()]
    string = ''
    for e in list(tmp.getdata()):
      string += str(e) + ' '
    row.append(string.rstrip())
    lists.append(row)
    # newx.append(list(tmp.getdata()))
    # newy.append(int(iy.tolist()))
    
    for j in range(1,6):
      # # tmp = tmp.rotate( j, PIL.Image.BILINEAR )
      # newx.append(list(tmp.rotate( 4*j, PIL.Image.BILINEAR ).getdata()))
      # newx.append(list(tmp.rotate( -4*j, PIL.Image.BILINEAR ).getdata()))
      # newy.append(int(iy.tolist()))
      # newy.append(int(iy.tolist()))
      row = [iy.tolist()]
      string = ''
      for e in list(tmp.rotate( 4*j, PIL.Image.BILINEAR ).getdata()):
        string += str(e) + ' '
      row.append(string.rstrip())
      lists.append(row)
      row = [iy.tolist()]
      string = ''
      for e in list(tmp.rotate( -4*j, PIL.Image.BILINEAR ).getdata()):
        string += str(e) + ' '
      row.append(string.rstrip())
      lists.append(row)
    tmp = PIL.Image.fromarray(np.uint8(tmp)).transpose(PIL.Image.FLIP_LEFT_RIGHT)
    row = [iy.tolist()]
    string = ''
    for e in list(tmp.getdata()):
      string += str(e) + ' '
    row.append(string.rstrip())
    lists.append(row)
    # newx.append(list(tmp.getdata()))
    # newy.append(int(iy.tolist()))
    for j in range(1,6):
      # newx.append(list(tmp.rotate( 4*j, PIL.Image.BILINEAR ).getdata()))
      # newx.append(list(tmp.rotate( -4*j, PIL.Image.BILINEAR ).getdata()))
      # newy.append(int(iy.tolist()))
      # newy.append(int(iy.tolist()))
      row = [iy.tolist()]
      string = ''
      for e in list(tmp.rotate( 4*j, PIL.Image.BILINEAR ).getdata()):
        string += str(e) + ' '
      row.append(string.rstrip())
      lists.append(row)
      row = [iy.tolist()]
      string = ''
      for e in list(tmp.rotate( -4*j, PIL.Image.BILINEAR ).getdata()):
        string += str(e) + ' '
      row.append(string.rstrip())
      lists.append(row)
    for r in lists:
      s.writerow(r)
    text.close()
  # for i in newx:
  #   plt.imshow(PIL.Image.fromarray(np.uint8(toPic(i))), cmap='gray', interpolation='nearest')
  #   plt.show()
  print(len(newx))
  # picsave(newx, newy)
  # x,y = readcsv('./ooo.csv')
  # for i, iy in zip(x, y):
  #   plt.imshow(toPic(i), cmap='gray', interpolation='nearest')
  #   plt.show()


