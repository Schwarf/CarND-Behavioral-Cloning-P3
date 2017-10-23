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
    Model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='valid'))
    Model.add(ELU())
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Dropout(0.5))
    
    Model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    Model.add(ELU())
    Model.add(MaxPooling2D(pool_size=(2, 2)))
    Model.add(Dropout(0.5))

    Model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode='valid'))
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

    Model.compile(optimizer =Adam(lr = 0.0001), loss= 'mse')
    
    history = Model.fit_generator(trainingGenerator, validation_data=validationGenerator, nb_val_samples=2560, samples_per_epoch=23040, nb_epoch=5, verbose=2)
    
     # Save model data
    Model.save('./model.h5')


def TheNvidiaModel(trainingGenerator, validationGenerator):
    print ("The modified Nvidia model")
    print("Keras Version: ", keras.__version__)
    Model = Sequential()
    dropoutRate=0.5
    #Model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    ### Three convolutional layers with ELU activation and 5x5 filter shapes and 2x2 strides
    # Input = 66x200x3

    

    Model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', input_shape = (66,200,3)))
    Model.add(ELU())
    #Model.add(MaxPooling2D(strides=(1,2)))
    Model.add(Dropout(dropoutRate))
    
    
    # Input = 31x98x24
    Model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid' ))
    Model.add(ELU())
    Model.add(Dropout(dropoutRate))
    
    # Input = 14x47x38
    Model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid' ))
    Model.add(ELU())
    Model.add(Dropout(dropoutRate))
    
    ### Two convolutional layers with ELU activation and 3x3 filter shapes and 1x1 strides
    # Input = 5x22x48
    Model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid' ))
    Model.add(ELU())
    Model.add(Dropout(dropoutRate))
    
    # Input = 3x20x64
    Model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    Model.add(ELU())
    Model.add(Dropout(dropoutRate))
    #Input = 64x1x18
    ### Flatten 
    Model.add(Flatten())
    
    #Input = 1152
    Model.add(Dense(100))
    Model.add(ELU())
    #Model.add(Dropout(dropoutRate))
    #Input = 100
    Model.add(Dense(50))
    Model.add(ELU())
    #Model.add(Dropout(0.5))
    #Input = 50
    Model.add(Dense(10))
    Model.add(ELU())
    #Model.add(Dropout(0.5))
    #Input = 10
    Model.add(Dense(1))
    print(Model.summary())
    
    #Compile Model with Adam optimizer and loss function equal to mean square error
    Model.compile(optimizer =Adam(lr = 0.0001), loss= 'mse')
    
    history = Model.fit_generator(trainingGenerator, validation_data=validationGenerator, nb_val_samples=2560, samples_per_epoch=23040, nb_epoch=7, verbose=2)
    
     # Save model data
    Model.save('./model.h5')
#    Model.save_weights('./model.h5')
#    json_string = Model.to_json()
#    with open('./model.json', 'w') as f:
#        f.write(json_string)