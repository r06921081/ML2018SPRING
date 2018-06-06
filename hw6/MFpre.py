from keras.models import load_model
from utils import *
import numpy as np
import sys

mean = 3.58171208604


testX = loadtest(sys.argv[1])
print(testX.shape)
model = load_model('./noupdate/0.85686/model.0.8507.h5', custom_objects={'rmse': normalize_rmse})

result = model.predict(np.hsplit(testX, 2), batch_size= 1024)
result = result * 1.116897661 + 3.581712086
print('--y_pred:\n', result)
# p([np.amax(result),np.amin(result)])
saveresult(sys.argv[2], result)