from keras.models import Sequential, load_model
import numpy as np
import dataprocess as dp
import sys

if sys.argv[3] == 'public':
    models = ['./first.h5', './ensemble.h5']
elif sys.argv[3] == 'private':
    models = ['./first.h5']
else:
    # give up ensamble in hw3 but have to do in hw4...
    models = [
        #'./noupdate/0.68904/first.h5', './noupdate/0.71022mistory/model.0279-0.7157.h5'#, './noupdate/0.68292/first.h5', './noupdate/0.68570/first.h5'
        #sys.argv[3]
        './1.h5','./2.h5','./4.h5','./7.h5','./8.h5','./10.h5','./0.71022mistory/model.0279-0.7157.h5'
        #'./model.0259-0.7937.h5','./0.71022mistory/model.0279-0.7157.h5'
    ]

X_test = dp.readtest(sys.argv[1])
X_nomean = X_test.copy()
X_nomean = X_nomean.reshape(-1, 48, 48, 1)/255.
X_test = X_test - np.mean(X_test)
X_test = X_test.reshape(-1, 48, 48, 1)/255.
result = []
# this is voting 
for i, mod in enumerate(models):
    model = load_model(mod)    
    if result == []:
        result = model.predict(X_test)
    else:
        if i == 6:
            result += model.predict(X_test)*5.8
        else:
            result += model.predict(X_test)
dp.savepre(result, sys.argv[2])
