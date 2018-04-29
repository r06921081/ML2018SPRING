import numpy as np
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input, Dropout, Reshape, Flatten, Conv2D, LeakyReLU, MaxPooling2D, UpSampling2D, Cropping2D, BatchNormalization
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
#import matplotlib.pyplot as plt
from sklearn.cluster import Birch
import sys
from keras.preprocessing.image import ImageDataGenerator

from tools import p
X = np.load('./data/image.npy')
# X = X - np.mean(X)
X = X/255.
x_train = X.reshape((X.shape[0], 28, 28,1))
b_size = 128


todim = 64

callback = [
            TensorBoard(),
            CSVLogger('./log.{epoch:04d}-{loss:.4f}.csv'),
            ModelCheckpoint('./model.{epoch:04d}-{loss:.4f}.h5', period=10),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=int(25), verbose=1),
            EarlyStopping(patience = 100)
            ]
if len(sys.argv) >= 2 and sys.argv[1] == 'continue':
    model = load_model(sys.argv[2])
else:
    input_size = Input(shape=(28,28,1))
    # encoder layers
    encoded = Conv2D(128, (5, 5), activation='relu', padding='same', name = 'c1')(input_size)
    # encoded = BatchNormalization()(encoded)
    encoded = MaxPooling2D((2, 2), padding='same', name = 'm1')(encoded)
    # encoded = Dropout(0.25)(encoded)
    encoded = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'c2')(encoded)
    # encoded = BatchNormalization()(encoded)
    encoded = MaxPooling2D((2, 2), padding='same', name = 'm2')(encoded)
    # encoded = Dropout(0.40)(encoded)
    encoded = Flatten()(encoded)
    encoded = Dense(1024, activation='relu', name = 'd1')(encoded)
    encoded = Dense(1024, activation='relu', name = 'd2')(encoded)
    encoder_output = Dense(600, activation='relu', name = 'd3')(encoded)
    decoded = Dense(1024, activation='relu', name = 'd4')(encoder_output)
    decoded = Dense(1024, activation='relu', name = 'd5')(decoded)
    decoded = Dense(6272, name = 'uf')(decoded)
    decoded = Reshape((7,7,128))(decoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(256, (3, 3), activation='relu', padding='same', name = 'c3')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(128, (5, 5), activation='relu', padding='same', name = 'c4')(decoded)
    decoded = Conv2D(1, (28, 28), activation='sigmoid', padding='same')(decoded)


    # construct the autoencoder model
    autoencoder = Model(inputs=input_size, outputs=decoded)
    # construct the encoder model for plotting
    encoder = Model(inputs=input_size, outputs=encoder_output)
    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()
    model = autoencoder
colors = ['r', 'b']
# training
train_history = model.fit(x_train, x_train,
                epochs=2000,
                batch_size=512,
                shuffle=True,
                callbacks=callback,
                )
# autoencoder.fit_generator(
#     generator = train_gen,
#     steps_per_epoch = train_gen.n // train_gen.batch_size,    
#     epochs =2000,
#     validation_data = train_gen,#(X_test/255., y_test),
#     validation_steps = 128,
#     use_multiprocessing = True,
#     callbacks=callback
# )
save_model(autoencoder, './new.h5')