
import numpy as np
import sys
np.random.seed(1126)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, Adagrad
from keras.layers.normalization import BatchNormalization 
from keras.preprocessing.image import ImageDataGenerator
import dataprocess as dp
import models as mod
from tools import p
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
(X_train, y_train) = dp.readcsv('./data/train.csv')



# feats, lables, _ = dp.read_dataset('./data/train.csv', shuffle = True)

validpart = int(X_train.shape[0]*0.2)
seed = np.random.randint(1,1126)
np.random.seed(seed)
np.random.shuffle(X_train)
np.random.seed(seed)
np.random.shuffle(y_train)
X_test = X_train[:validpart].reshape(-1, 48, 48, 1)#/255.
X_train = X_train[validpart:].reshape(-1, 48, 48, 1)#/255.
y_test = np_utils.to_categorical(y_train[:validpart], num_classes=7)
y_train = np_utils.to_categorical(y_train[validpart:], num_classes=7)
callback = [
            TensorBoard(),
            CSVLogger('./log.csv', append=True),
            ModelCheckpoint('./model.{epoch:04d}-{val_acc:.4f}.h5', period=10),
            ReduceLROnPlateau('val_loss', factor=0.1, patience=int(100/4), verbose=1),
            # EarlyStopping(patience = 100)
            ]
# data pre-processing
b_size = 300
train_data_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
train_data_gen.fit(X_train)
train_gen = train_data_gen.flow(X_train, y_train, batch_size = b_size)
val_data_gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
)
val_data_gen.fit(X_test)
val_gen = val_data_gen.flow(X_test, y_test, batch_size = b_size)

# Another way to build your CNN
# model = mod.build_ta_model()
# model = mod.res50((None, 48, 48, 1))
if len(sys.argv) > 1:
    if sys.argv[1] == 'continue':
        model = load_model(sys.argv[2])
else:
        # model = mod.ll((None, 48, 48, 1))
    model = mod.mygoodgoodmodel((None, 48, 48, 1))
        # model = mod.res50((None, 48, 48, 1))

print('Training ------------')
# Another way to train the model
# p(X_train.shape)
model.fit_generator(
    generator = train_gen,
    steps_per_epoch = train_gen.n // train_gen.batch_size,    
    epochs =2000,
    validation_data = val_gen,
    validation_steps = X_test.shape [0] // train_gen.batch_size ,
    use_multiprocessing = True,
    callbacks=callback
)
# model.fit(X_train, y_train, epochs=140, batch_size=128, callbacks=callback, validation_data = (X_test, y_test))
model.save('./first.h5')
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

X_test = dp.readtest('./data/test.csv')
X_test = X_test.reshape(-1, 48, 48, 1)/255.
result = model.predict(X_test)
dp.savepre(result, './result/oo.csv')
print(result)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
