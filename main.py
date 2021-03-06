'''
Created on Oct 22, 2017

@author: andre
'''
from data import *
from model import *




from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

csvData = LoadCSVFiles()

averagePerBin = DataVisualization(csvData['Steering'].tolist(), "Original data")
csvData = DataPreparation(csvData, averagePerBin)
DataVisualization(csvData['Steering'].tolist(), "Cleansed data")
trainingData, validationData = SplitDataInTrainingAndValidationSet(csvData)
print("Number of data sets in training data ", len(trainingData))
print("Number of data sets in validation data ", len(validationData))

Model = 'Other'
Model = 'Nvidia1'
trainingGenerator = DataGenerator(trainingData, Model, batchSize =128)
validationGenerator = DataGenerator(validationData, Model, batchSize =128)


Models = ["NvidiaOriginal", "NvidiaWithActivation", "NvidiaWithDropout"]

#DataVisualization(newSet)
if(Model == 'Nvidia1'):
    model.TheWorkingNvidiaMod(trainingGenerator, validationGenerator)
elif(Model == 'Nvidia2'):
    model.TheNvidiaModel2(trainingGenerator, validationGenerator)

elif(Model == 'Other'):
    model.OtherModel(trainingGenerator, validationGenerator)

