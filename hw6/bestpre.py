from keras.models import load_model
from utils import *
import numpy as np
import sys

mean = 3.58171208604


testX = loadtest(sys.argv[1])
movie = loadmovie(sys.argv[3])
user = loaduser(sys.argv[4])
# p(user)
X_movieYear = np.array([movie[movieId][0] for movieId in testX[:,1]])
X_movieType = np.array([movie[movieId][1:] for movieId in testX[:,1]])
X_user = np.array([user[userId] for userId in testX[:,0]])
print(testX.shape)
model = load_model('./512Bmodel.0.8448.h5', custom_objects={'normalize_rmse': normalize_rmse})

result = model.predict(np.hsplit(testX, 2) + [X_movieYear,X_movieType], batch_size= 1024)
result = result * 1.116897661 + 3.581712086
print('--y_pred:\n', result)
# p([np.amax(result),np.amin(result)])
saveresult(sys.argv[2], result)