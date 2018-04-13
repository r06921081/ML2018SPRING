from keras.models import Sequential, load_model
import dataprocess as dp
import sys
models = [
    # './noupdate/0.68292/first.h5', './noupdate/0.68904/first.h5', './noupdate/0.68570/first.h5'
    sys.argv[1]
]
X_test = dp.readtest('./data/test.csv')
X_test = X_test.reshape(-1, 48, 48, 1)/255.
result = []
for mod in models:
    model = load_model(mod)    
    if result == []:
        result = model.predict(X_test)
    else:
        result += model.predict(X_test)
dp.savepre(result, './result/oo.csv')