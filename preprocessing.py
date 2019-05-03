import os
import nltk
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import sentiwordnet as swn
from nltk.stem.porter import *
'''
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
'''
stemmer = PorterStemmer()

negativeFileList = os.listdir('txt_sentoken/neg')
positiveFileList = os.listdir('txt_sentoken/pos')

negativeText = []
positiveText = []

doStem = True
removeStopWords = True

def loadNRCLexicon():
    NRCDict = {}
    with open('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')as f:
        dictText = f.read()
        dictText =dictText.split('\n')
        dictText.pop(0)
        dictText.pop(-1)
        for i in range( int(len(dictText)/10) ):
            temp = []
            word = None
            for j in range( i*10, (i+1)*10 ):
                lex = dictText[j].split('\t')
                word = lex[0]
                temp.append( int(lex[-1]))
            NRCDict[ word ] =  np.array(temp)
    return NRCDict

NRCDict = loadNRCLexicon()

def preprocessing(path, doStem = False, removeStopWords = True):
    tagTranslate =  {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'}
    flist = os.listdir(path)
    resList = []
    posList = []
    for fname in flist:
        fullPath = path + fname
        f = open(fullPath)
        text = word_tokenize(f.read())
        f.close()
        #if (doStem):
            #text = [stemmer.stem(word) for word in text]
        # if(removeStopWords):
        resList.append(text)
        posTags =  nltk.pos_tag(text)
        temp = []
        for ind,tag in enumerate(posTags):
            if( tag[1][:2] in tagTranslate):
                temp.append( tagTranslate[ tag[1][:2] ])
            else:
                temp.append('')
        posList.append(temp)
    return resList, posList

def getScore(word, pos):
    temp = word + '.' + pos + '.01'
    try:
        return (swn.senti_synset(temp).pos_score(), swn.senti_synset(temp).neg_score())
    except:
        return (0,0)
    #return

def getSumOfScore(text, postag):
    scoreList = []
    pos = 0
    neg = 0
    for ind, word in enumerate(text):
        res = getScore(word, postag[ind])
        #print(res)
        scoreList.append( res[0] )
        scoreList.append( -res[1] )
        pos += res[0]
        neg += res[1]
    scoreList.sort(key = lambda x:-abs(x) )
    return (pos,neg), scoreList[:7]

def getSumOfNRCScore(text):
    res = np.zeros((10,))
    for word in text:
        try:
            res += NRCDict[word]
        except:
            continue
    return res

negativeText = None
positiveText = None
negativePos = None
positivePos = None
try:
    with open('processed_neg') as f:
        negativeText = json.loads(f.read())
    with open('processed_pos') as f:
        positiveText = json.loads(f.read())
    with open('neg_pos') as f:
        negativePos = json.loads(f.read())
    with open('pos_pos') as f:
        positivePos = json.loads(f.read())
except:
    negativeText,negativePos = preprocessing('txt_sentoken/neg/')
    positiveText,positivePos = preprocessing('txt_sentoken/pos/')
    with open('processed_neg','w') as f:
        f.write( json.dumps(negativeText))

    with open('neg_pos','w') as f:
        f.write( json.dumps(negativePos))

    with open('processed_pos','w') as f:
        f.write( json.dumps(positiveText))

    with open('pos_pos','w') as f:
        f.write( json.dumps(positivePos))

def generateTrainingData(text,pos, isPos = True):
    featureMatrix = []
    labels = []
    for ind in range(len(text)):
        features = []
        emotionScore, highestEmotionScores = getSumOfScore(text[ind], pos[ind])
        NRCScore = getSumOfNRCScore(text[ind])
        features = list(NRCScore) + list(highestEmotionScores) + list(emotionScore) + list([len(text[ind])])
        featureMatrix.append( np.array(features))
        if(isPos):
            labels.append( 1 )
        else:
            labels.append( 0 )
    return featureMatrix, labels

pos_features, pos_labels = generateTrainingData(positiveText, positivePos)
neg_features, neg_labels = generateTrainingData(negativeText, negativePos, False)

features = np.array( pos_features + neg_features )
labels = np.array( pos_labels + neg_labels )

np.save('features.npy', features)
np.save('labels.npy', labels)