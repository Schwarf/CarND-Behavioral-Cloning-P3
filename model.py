'''
Created on Oct 20, 2017

@author: andre
'''
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
import keras




def OtherModel(trainingGenerator, validationGenerator):
    print ("The other model")
    print("Keras Version: ", keras.__version__)
    Model = Sequential()
    
    Model.add(Convolution2D(3, 1, 1, subsample=(1, 1), border_mode='valid', input_shape=(32,32,3)))
    Model.add(Convolution2D(32, 7, 7, subsample=(1, 1), border_mode='valid'))
    Model.add(ELU())
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Dropout(0.5))
    
    Model.add(Convolution2D(64, 7, 7, subsample=(1, 1), border_mode='valid'))
    Model.add(ELU())
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Dropout(0.5))

    Model.add(Convolution2D(128, 1, 1, subsample=(1, 1), border_mode='valid'))
    Model.add(ELU())
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Dropout(0.5))
    
    Model.add(Flatten())
    
    Model.add(Dense(512))
    Model.add(ELU())
    Model.add(Dense(64))
    Model.add(ELU())
    Model.add(Dense(16))
    Model.add(Activation('relu'))

    Model.add(Dense(1))
    print(Model.summary())
    Model.compile(optimizer =Adam(lr = 0.0001), loss= 'mse')
    
    history = Model.fit_generator(trainingGenerator, validation_data=validationGenerator, nb_val_samples=2560, samples_per_epoch=25600, nb_epoch=10, verbose=2)
    
     # Save model data
    Model.save('./model.h5')




def TheNvidiaModel2(trainingGenerator, validationGenerator):
    print ("The second modified Nvidia model")
    print("Keras Version: ", keras.__version__)
    dropoutRate = 0.5
    Model = Sequential()

    Model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', input_shape=(66,200,3) ))
    Model.add(Activation('relu'))
    Model.add(Dropout(dropoutRate))
    
    Model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid' ))
    Model.add(Activation('relu'))
    Model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid' ))
    Model.add(Activation('relu'))
    Model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same'))
    Model.add(Activation('relu'))
    Model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid'))
    Model.add(Activation('relu'))
    Model.add(Dropout(dropoutRate))
    
    Model.add(Flatten())
    
    Model.add(Dense(1000))
    
    Model.add(Dense(100))
    
    Model.add(Dense(10))
    
    Model.add(Dense(1))
    
    print(Model.summary())
    
    Model.compile(optimizer =Adam(lr = 0.0001), loss= 'mse')
    
    history = Model.fit_generator(trainingGenerator, validation_data=validationGenerator, nb_val_samples=2560, samples_per_epoch=20480, nb_epoch=10, verbose=2)
    
    Model.save('./model.h5')


def TheNvidiaModel1(trainingGenerator, validationGenerator):
    print ("The first modified Nvidia model, hits the curb in the right curve.")
    print("Keras Version: ", keras.__version__)
    dropoutRate = 0.5
    Model = Sequential()

    Model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', input_shape=(66,200,3) ))
    Model.add(Activation('relu'))
    Model.add(Dropout(dropoutRate))
    
    Model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid' ))
    Model.add(Activation('relu'))
    Model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid' ))
    Model.add(Activation('relu'))
    Model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='same'))
    Model.add(Activation('relu'))
    Model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid'))
    Model.add(Activation('relu'))
    
    Model.add(Flatten())
    
    Model.add(Dense(1000))
    
    Model.add(Dense(100))
    
    Model.add(Dense(10))
    
    Model.add(Dense(1))
    
    print(Model.summary())
    
    Model.compile(optimizer =Adam(lr = 0.0001), loss= 'mse')
    
    history = Model.fit_generator(trainingGenerator, validation_data=validationGenerator, nb_val_samples=2560, samples_per_epoch=20480, nb_epoch=10, verbose=2)
    
    Model.save('./model.h5')
