from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import nltk
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
nltk.download('stopwords')
import json
import numpy as np
posList = None
negList = None
vecSize = 100
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda,Activation, Concatenate, LSTM, Masking
from keras.layers.merge import Add, Multiply, multiply
from keras.optimizers import Adam
stopWords = set(stopwords.words('english'))
maxLen = 800
def prepRNN():
    model = load_model('rnn2.model')
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))
    return model
    input = Input(shape=(maxLen, vecSize))
    #masked_state = Masking(mask_value=-5000)(input)
    memory_layer = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(input)
    memory_layer_2 = Dense(256, activation='relu')(memory_layer)
    output = Dense(1, activation='sigmoid')(memory_layer_2)
    model = Model( input = input, output = output)
    model.compile(loss = 'binary_crossentropy', metrics = ['acc'], optimizer= Adam(0.0001) )
    return model

def prepareVec():
    model = None
    try:
        model = Word2Vec.load('reviewVec.model')
    except:
        model = Word2Vec( common_texts + posList + negList, size=vecSize, window=5, min_count=3, workers=4)
        model.save("reviewVec.model")

    return model


cnt = 0
def text2Vec(texts, w2v):
    global  maxLen, cnt
    res = []
    for text in texts:
        #maxLen = max( maxLen, len(text))
        res_per_text = []
        for word in text:
            if(word in stopWords): continue
            try:
                res_per_text.append( w2v.wv[word] )
            except:
                #res_per_text.append( np.zeros((vecSize,)) )
                continue
        if (len(res_per_text) > 640): cnt += 1
        res.append( np.array(res_per_text) )
    res = np.array(res)
    return res

with open('processed_neg') as f:
    t = f.read()
    t = json.loads(t)
    negList = t
with open('processed_pos') as f:
    t = f.read()
    t = json.loads(t)
    posList = t

vecModel = prepareVec()
posFeature = text2Vec(posList, vecModel)
negFeature = text2Vec(negList, vecModel)
labels = np.zeros( (len(posFeature) + len(negFeature), ) )
labels[ : int(len(labels)/2) ]+=1

posFeature = keras.preprocessing.sequence.pad_sequences(posFeature, maxLen, 'float32',padding ='post',truncating='post', value =0)
negFeature = keras.preprocessing.sequence.pad_sequences(negFeature, maxLen, 'float32',padding ='post',truncating='post', value =0)
training_features = np.concatenate((posFeature[:900],negFeature[:900]))
testing_features = np.concatenate( (posFeature[900:], negFeature[900:]))
training_labels = np.concatenate( ( np.zeros((900,)) +1 , np.zeros((900,)) ) )
testing_labels = np.concatenate( ( np.zeros(100,)+1, np.zeros(100,) ) )
model = prepRNN()
model.save('rnn2.model')
es = EarlyStopping(min_delta=0.00002, patience=450,monitor='val_loss', restore_best_weights=True)
model.fit(  training_features, training_labels, epochs = 2000, batch_size= 25, validation_data=(testing_features, testing_labels), initial_epoch=1270,callbacks=[es])
model.save('rnn2.model')
#print('haha')