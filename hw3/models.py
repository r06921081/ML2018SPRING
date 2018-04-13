from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, ZeroPadding2D, Conv2D, Input, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import Adam, Adagrad, SGD
from keras import layers
from keras.models import Sequential, Model
from tools import p
def tmpmodel(inputsize):
  model = Sequential()

  model.add(Convolution2D(
      batch_input_shape=inputsize, filters=64, kernel_size=3,
      strides=1, padding='same'))
  model.add(Activation('relu'))
  # model.add(Dropout(0.10))
  model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='same'))
  model.add(Activation('relu'))
  # model.add(Dropout(0.10))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.10))

  # model.add(Convolution2D(filters=128, kernel_size=3, strides=1, padding='same'))
  model.add(BatchNormalization())
  # model.add(Activation('relu'))
  # model.add(Dropout(0.25))
  model.add(Convolution2D(filters=128, kernel_size=3, strides=1, padding='same'))
  model.add(Activation('relu'))
  # model.add(Dropout(0.25))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.25))

  # model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same'))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.35))
  # model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same'))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.35))
  model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  # model.add(Dropout(0.35))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.35))

  # model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.4))
  # model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.4))
  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(Activation('relu'))
  # model.add(Dropout(0.4))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.4))

  # model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.45))
  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(Activation('relu'))
  # model.add(Dropout(0.45))
  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(Activation('softplus'))
  # model.add(Dropout(0.45))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.45))

  model.add(Flatten())
  model.add(Dense(4096, activation = 'relu'))
  # model.add(Dropout(0.5))
  model.add(Dense(4096, activation = 'softplus'))
  model.add(Dropout(0.5))
  model.add(Dense(7))
  model.add(Activation('softmax'))

  # Another way to define your optimizer
  adam = Adam(lr=1e-4)
  sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

  # We add metrics to get more results you want to see
  model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.summary()
  return model

def build_dnn(dropout = 0.2, nb_class = 7):
    model  = Sequential()

    model.add(Flatten(input_shape = (48, 48, 1)))

    model.add(Dense(9261))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(4608))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.35))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('softplus'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_class))  
    model.add(Activation('softmax'))


    model.compile(loss = 'categorical_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy'])
    model.summary()

    return model

def little(inputsize):
  model = Sequential()

  model.add(Convolution2D(batch_input_shape=inputsize,
      filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer='random_uniform'))
  model.add(Activation('relu'))
  model.add(Convolution2D(filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer='random_uniform'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.10))

  model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='random_uniform'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.25))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.25))

  model.add(Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='random_uniform'))
  model.add(Activation('relu'))
  model.add(Dropout(0.35))
  model.add(Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='random_uniform'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.35))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', kernel_initializer='random_uniform'))
  model.add(Dropout(0.35))

  model.add(Convolution2D(filters=256, kernel_size=5, strides=1, padding='same', kernel_initializer='random_uniform'))
  model.add(Activation('relu'))
  model.add(Dropout(0.4))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', kernel_initializer='random_uniform'))
  model.add(Dropout(0.4))

  model.add(Convolution2D(filters=256, kernel_size=5, strides=1, padding='same', kernel_initializer='random_uniform'))
  model.add(Activation('relu'))
  model.add(Dropout(0.45))
  model.add(Convolution2D(filters=256, kernel_size=7, strides=1, padding='same', kernel_initializer='random_uniform'))
  model.add(Activation('softplus'))
  model.add(Dropout(0.45))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.45))

  model.add(Flatten())
  model.add(Dense(4096, activation = 'relu', kernel_initializer='random_uniform'))
  model.add(Dropout(0.5))
  model.add(Dense(4096, activation = 'softplus', kernel_initializer='random_uniform'))
  model.add(Dropout(0.5))
  model.add(Dense(7))
  model.add(Activation('softmax'))

  # Another way to define your optimizer
  adam = Adam(lr=1e-4)

  # We add metrics to get more results you want to see
  model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.summary()
  return model

def mygoodgoodmodel(inputsize):
  model = Sequential()

  model.add(Convolution2D(batch_input_shape=inputsize,
      filters=64, kernel_size=3, strides=1, padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.10))
  model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(PReLU(init='zero', weights=None))
  model.add(Dropout(0.10))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.10))

  model.add(Convolution2D(filters=128, kernel_size=3, strides=1, padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.25))
  model.add(Convolution2D(filters=128, kernel_size=3, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(PReLU(init='zero', weights=None))
  model.add(Dropout(0.25))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.25))

  model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.35))
  model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.35))
  model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(PReLU(init='zero', weights=None))
  model.add(Dropout(0.35))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.35))

  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.4))
  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.4))
  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(PReLU(init='zero', weights=None))
  model.add(Dropout(0.4))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.4))

  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(LeakyReLU(alpha=0.1))
  model.add(Dropout(0.45))
  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(PReLU(init='zero', weights=None))
  model.add(Dropout(0.45))
  model.add(Convolution2D(filters=512, kernel_size=3, strides=1, padding='same'))
  model.add(BatchNormalization())
  model.add(Activation('softplus'))
  model.add(Dropout(0.45))
  model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
  model.add(Dropout(0.45))

  model.add(Flatten())
  model.add(Dense(4096, activation = 'relu'))
  model.add(Dropout(0.4))
  model.add(Dense(4096, activation = 'softplus'))
  model.add(Dropout(0.4))
  model.add(Dense(7))
  model.add(Activation('softmax'))

  # Another way to define your optimizer
  adam = Adam(lr=1e-4)

  # We add metrics to get more results you want to see
  model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.summary()
  return model

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), drop = 0.0):
  filters1, filters2, filters3 = filters

  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(filters1, (1, 1), strides=strides,
              name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)
  x = Dropout(drop)(x)

  x = Conv2D(filters2, kernel_size, padding='same',
              name=conv_name_base + '2b')(x)
  x = BatchNormalization(name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)
  x = Dropout(drop)(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(name=bn_name_base + '2c')(x)

  shortcut = Conv2D(filters3, (1, 1), strides=strides,
                    name=conv_name_base + '1')(input_tensor)
  shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
  shortcut = Dropout(drop)(shortcut)

  x = layers.add([x, shortcut])
  x = Activation('relu')(x)
  x = Dropout(drop)(x)
  return x

def identity_block(input_tensor, kernel_size, filters, stage, block, drop = 0.0):
  filters1, filters2, filters3 = filters

  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
  x = BatchNormalization(name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)
  x = Dropout(drop)(x)

  x = Conv2D(filters2, kernel_size,
              padding='same', name=conv_name_base + '2b')(x)
  x = BatchNormalization(name=bn_name_base + '2b')(x)
  x = Activation('relu')(x)
  x = Dropout(drop)(x)

  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
  x = BatchNormalization(name=bn_name_base + '2c')(x)

  x = layers.add([x, input_tensor])
  x = Activation('relu')(x)
  x = Dropout(drop)(x)
  return x


def res50(inputsize):
  bn_axis = 3 # channel last
  img_input = Input((48, 48, 1))
  x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
  x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), drop=0.1)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', drop=0.1)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', drop=0.1)

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', drop=0.15)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', drop=0.15)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', drop=0.15)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', drop=0.15)

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', drop=0.25)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', drop=0.25)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', drop=0.25)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', drop=0.25)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', drop=0.25)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', drop=0.25)

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', drop=0.35)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', drop=0.35)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', drop=0.35)

  # x = GlobalAveragePooling2D()(x)
  # output = Activation('softmax',name='predictions')(x)

  x = Flatten()(x)
  output = Dense(2306, activation='relu', name='fc1')(x)
  output = Dense(7, activation='softmax', name='fc7')(x)
  model = Model(img_input, output, name='resnet50')
  adam = Adam(lr=1e-4)

  # We add metrics to get more results you want to see
  model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.summary()
  return model

def build_ta_model(input_shape = (48, 48, 1), num_classes = 7):
    input_img = Input(input_shape)
    
    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)
    block2 = Dropout(0.1)(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)
    block3 = Dropout(0.2)(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)
    block4 = Dropout(0.3)(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Dropout(0.4)(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(num_classes)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def build_dnn(dropout = 0.2, nb_class = 7):
    model  = Sequential()

    model.add(Flatten(input_shape = (48, 48, 1)))

    model.add(Dense(2306))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Dense(4612))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2306))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('softplus'))
    model.add(Dropout(0.4))

    model.add(Dense(nb_class))  
    model.add(Activation('softmax'))


    model.compile(loss = 'categorical_crossentropy', 
                optimizer = 'adam', 
                metrics = ['accuracy'])
    model.summary()

    return model

def ll(inputsize):
  fashion_model = Sequential()
  fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(48,48,1)))
  fashion_model.add(LeakyReLU(alpha=0.1))
  fashion_model.add(MaxPooling2D((2, 2),padding='same'))
  fashion_model.add(Dropout(0.25))
  fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
  fashion_model.add(LeakyReLU(alpha=0.1))
  fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
  fashion_model.add(Dropout(0.25))
  fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
  fashion_model.add(LeakyReLU(alpha=0.1))                  
  fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
  fashion_model.add(Dropout(0.4))
  fashion_model.add(Flatten())
  fashion_model.add(Dense(128, activation='linear'))
  fashion_model.add(LeakyReLU(alpha=0.1))           
  fashion_model.add(Dropout(0.3))
  fashion_model.add(Dense(7, activation='softmax'))

  # Another way to define your optimizer
  adam = Adam()
  sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)

  # We add metrics to get more results you want to see
  fashion_model.compile(optimizer=adam,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  fashion_model.summary()
  

  return fashion_model