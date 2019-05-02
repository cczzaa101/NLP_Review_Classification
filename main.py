import os
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()

negativeFileList = os.listdir('txt_sentoken/neg')
positiveFileList = os.listdir('txt_sentoken/pos')

negativeText = []
positiveText = []

doStem = True
removeStopWords = True

def preprocessing(path, doStem = False, removeStopWords = True):
    flist = os.listdir(path)
    resList = []
    for fname in flist:
        fullPath = path + fname
        f = open(fullPath)
        text = word_tokenize(f.read())
        f.close()
        if (doStem):
            text = [stemmer.stem(word) for word in text]
        # if(removeStopWords):
        resList.append(text)
    return resList

negativeText = None
positiveText = None
try:
    with open('processed_neg') as f:
        negativeText = json.loads(f.read())
    with open('processed_pos') as f:
        positiveText = json.loads(f.read())
except:
    negativeText = preprocessing('txt_sentoken/neg/')
    positiveText = preprocessing('txt_sentoken/pos/')
    with open('processed_neg','w') as f:
        f.write( json.dumps(negativeText))

    with open('processed_pos','w') as f:
        f.write( json.dumps(positiveText))

print('gaga')