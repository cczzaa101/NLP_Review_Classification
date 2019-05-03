import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import Adam
import numpy as np

features = np.load('features.npy')
labels = np.load('labels.npy')

def splitForValidation(slice):
    trainingFeatures = []
    trainingLabels = []
    testingFeatures = []
    testingLabels = []
    for ind in range( len(features) ):
        if( ind %5 == slice ):
            testingLabels.append( labels[ind] )
            testingFeatures.append(features[ind])
        else:
            trainingFeatures.append( features[ind] )
            trainingLabels.append( labels[ind] )
    return (trainingFeatures, trainingLabels), (testingFeatures,testingLabels)

(trainingSet, trainingLabels), (testingSet, testingLabels) = splitForValidation(1)

model = Sequential()
model.add( Dense(128, input_dim = len(features[0]), activation= 'relu' ) )
model.add(Dropout(0.5))
model.add( Dense(128, input_dim = len(features[0]), activation= 'relu' ) )
model.add(Dropout(0.5))
model.add( Dense(1, activation= 'sigmoid' ) )
model.compile(optimizer= Adam(0.00005),metrics=['accuracy'], loss='binary_crossentropy')

model.fit( np.array(trainingSet), np.array(trainingLabels), epochs = 2000, batch_size= 100 )

def evaluate(testingSet, testingLabels):
    res = model.predict( np.array(testingSet) )
    correctCount = 0
    totalCount = 0
    for i,value in enumerate(res):
        if(value>0.5): res[i] = 1
        else: res[i] = 0
        if(res[i] == testingLabels[i]):
            correctCount+= 1
        totalCount+=1

    print( correctCount/ totalCount )
    #return res

evaluate(testingSet, testingLabels)
#score = model.evaluate( np.array(testingSet), np.array(testingLabels) )
#print(score)