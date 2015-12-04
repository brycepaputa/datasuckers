#!/usr/bin/env python2

__author__ = 'bryce'
__date__ = '$Nov 5, 2015 7:07:51 PM$'

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from basicLogisticRegression import *
import random

def shuffle(df1, df2=None):
    index = list(df1.index)
    random.shuffle(index)
    df1 = df1.ix[index]
    df1.reset_index()
    if df2 != None:
        df2 = df2.ix[index]
        df2.reset_index()
        return df1, df2
    return df1

def indicatorMap(x, value):
    return (1 if str(x).lower() == str(value).lower() else 0)

def wordScaleMap(x, scale):
    return scale.index(str(x).lower());

gradeScale = ['F', 'D-', 'D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
def gradeNumMap(x):
    if x in gradeScale:
        return gradeScale.index(x)+1
    return 0

def fixIndices(data):#xtrain, ytrain, xtest):    
#    if xtest.index[0] == 1:
#        xtest.index = xtest.index - 1
#    if xtrain.index[0] == 1:
#        xtrain.index = xtrain.index - 1
#    if ytrain.index[0] == 1:
#        ytrain.index = ytrain.index - 1
#    return xtrain, ytrain, xtest
    for i in range(len(data)):
        if data[i].index[0] > 0:
            data[i].index = data[i].index - data[i].index[0]
    return data

def cleanData(xtrain, ytrain, xvalid, yvalid, xtest, keepQuality=False):
    #['id', 'tid', 'dept', 'date', 'forcredit', 'attendance',
    #       'textbookuse', 'interest', 'grade', 'tags', 'comments', 'helpcount',
    #       'nothelpcount', 'online', 'profgender', 'profhotness',
    #       'helpfulness', 'clarity', 'easiness', 'quality']
    indicatorColumns = ['attendance', 'dept', 'forcredit']
    wordScaleColumns = {
        'textbookuse': [
            'nan', 
            'what textbook?', 
            'barely cracked it open', 
            'you need it sometimes', 
            "it's a must have", 
            'essential to passing'
        ], 
        'interest': [
            'nan',
            'low',
            'meh',
            'sorta interested',
            'really into it',
            "it's my life"
        ],
        'online': [
            'nan',
            'online'
        ]
    }
    
    for col in indicatorColumns:
        print col
        values = xtrain[col].unique();
        for val in values:
            xtrain[col+str(val)] = xtrain[col].map(lambda x: indicatorMap(x, val))
            xvalid[col+str(val)] = xvalid[col].map(lambda x: indicatorMap(x, val))
            xtest[col+str(val)] = xtest[col].map(lambda x: indicatorMap(x, val))
        xtrain.drop(col, axis=1, inplace=True)
        xvalid.drop(col, axis=1, inplace=True)
        xtest.drop(col, axis=1, inplace=True)

    for col in wordScaleColumns.keys():
        print col
        xtrain[col] = xtrain[col].map(lambda x: wordScaleMap(x, wordScaleColumns[col]))
        xvalid[col] = xvalid[col].map(lambda x: wordScaleMap(x, wordScaleColumns[col]))
        xtest[col] = xtest[col].map(lambda x: wordScaleMap(x, wordScaleColumns[col]))
        
    tags = ['Assignments galore', 'Participation matters', 'Tests? Not many', 'Hilarious', 'Tests are tough', 'Pop quiz master', 'Gives good feedback', 'Clear grading criteria', 'Tough Grader', 'Get ready to read', 'Better Like Group Projects', 'Amazing lectures', 'Inspirational', 'Respected by students', 'There for you', 'Big time extra credit', 'Lectures are long', "Skip class? You won't pass.", 'Papers? More like novels', 'Would take again']
    
    for tag in tags:
        xtrain['tags'+tag] = 0
        xvalid['tags'+tag] = 0
        xtest['tags'+tag] = 0
    
    for row in range(0, len(xtrain)):
        current = xtrain.tags[row].replace('[', '').replace(']', '').split(', ')
        for tag in current:
            xtrain.set_value(row, 'tags'+tag[1:len(tag)-1], 1)
    for row in range(0, len(xvalid)):
        current = xvalid.tags[row].replace('[', '').replace(']', '').split(', ')
        for tag in current:
            xvalid.set_value(row, 'tags'+tag[1:len(tag)-1], 1)
    for row in range(0, len(xtest)):
        current = xtest.tags[row].replace('[', '').replace(']', '').split(', ')
        for tag in current:
            xtest.set_value(row, 'tags'+tag[1:len(tag)-1], 1)
    
    xtrain.drop('tags', axis=1, inplace=True)
    xvalid.drop('tags', axis=1, inplace=True)
    xtest.drop('tags', axis=1, inplace=True)
    
    gradeValues =  [val for val in xtrain.grade.unique() if not(val in gradeScale)]
    for val in gradeValues:
        xtrain['grade'+str(val)] = xtrain['grade'].map(lambda x: indicatorMap(x, val))
        xvalid['grade'+str(val)] = xvalid['grade'].map(lambda x: indicatorMap(x, val))
        xtest['grade'+str(val)] = xtest['grade'].map(lambda x: indicatorMap(x, val))
    xtrain['gradenan'] = xtrain['grade'].map(lambda x: indicatorMap(x, 'nan'))
    xvalid['gradenan'] = xvalid['grade'].map(lambda x: indicatorMap(x, 'nan'))
    xtest['gradenan'] = xtest['grade'].map(lambda x: indicatorMap(x, 'nan'))
    xtrain['grade'] = xtrain['grade'].map(gradeNumMap)
    xvalid['grade'] = xvalid['grade'].map(gradeNumMap)
    xtest['grade'] = xtest['grade'].map(gradeNumMap)
    print 'done'
    xtrain.to_csv('xtrain_clean.csv')
    ytrain.to_csv('ytrain_clean.csv')
    xvalid.to_csv('xvalid_clean.csv')
    yvalid.to_csv('yvalid_clean.csv')
    xtest.to_csv('xtest_clean.csv')
    if not keepQuality:
        xtrain.drop(['helpfulness', 'clarity', 'easiness', 'quality'], axis=1, inplace=True)
        xvalid.drop(['helpfulness', 'clarity', 'easiness', 'quality'], axis=1, inplace=True)
    return xtrain, ytrain, xvalid, yvalid, xtest

def vectorizeWords(xtrain, xvalid, xtest):    
    countVect = CountVectorizer(min_df=120, stop_words='english', ngram_range=(1, 2))
    
    xtrWords = countVect.fit_transform(xtrain.comments.fillna(''))
    xvaWords = countVect.transform(xvalid.comments.fillna(''))
    xteWords = countVect.transform(xtest.comments.fillna(''))
    
    columns = [col.upper() for col in countVect.get_feature_names()]
    
    xtrDFWords = pd.DataFrame(xtrWords.A, columns=columns)
    xvaDFWords = pd.DataFrame(xvaWords.A, columns=columns)
    xteDFWords = pd.DataFrame(xteWords.A, columns=columns)
    
    xtrain = xtrain.drop('comments', axis=1)
    xvalid = xvalid.drop('comments', axis=1)
    xtest = xtest.drop('comments', axis=1)
    
    xtrain = pd.concat([xtrain, xtrDFWords], axis=1)
    xvalid = pd.concat([xvalid, xvaDFWords], axis=1)
    xtest = pd.concat([xtest, xteDFWords], axis=1)
    
    return (xtrain, xvalid, xtest, countVect)

def readCleanData(keepQuality=False):
    xtrain = pd.read_csv('xtrain_clean.csv')
    ytrain = pd.read_csv('ytrain_clean.csv', header=None, index_col=0)
    xvalid = pd.read_csv('xvalid_clean.csv')
    yvalid = pd.read_csv('yvalid_clean.csv', header=None, index_col=0)
    if not keepQuality:
        xtrain.drop(['helpfulness', 'clarity', 'easiness', 'quality'], axis=1, inplace=True)
        xvalid.drop(['helpfulness', 'clarity', 'easiness', 'quality'], axis=1, inplace=True)
    ytrain = ytrain[ytrain.columns[0]]
    yvalid = yvalid[yvalid.columns[0]]
    xtest = pd.read_csv('xtest_clean.csv')
    (xtrain, ytrain, xvalid, yvalid, xtest) = fixIndices((xtrain, ytrain, xvalid, yvalid, xtest))
    return xtrain, ytrain, xvalid, yvalid, xtest

def readData():
    xtrain = pd.read_csv('train.csv')
    shuffle(xtrain)
    xtest = pd.read_csv('test.csv')
    shuffle(xtest)
    ytrain = xtrain.quality-2
    nvalid = len(xtrain)/4
    xvalid = xtrain[0:nvalid]
    xtrain = xtrain[nvalid:]
    yvalid = ytrain[0:nvalid]
    ytrain = ytrain[nvalid:]
    (xtrain, ytrain, xvalid, yvalid, xtest) = fixIndices((xtrain, ytrain, xvalid, yvalid, xtest))
    return xtrain, ytrain, xvalid, yvalid, xtest
    

def main():
    xtrain, ytrain, xvalid, yvalid, xtest = readData()
    xtrain, ytrain, xvalid, yvalid, xtest = cleanData(xtrain, ytrain, xvalid, yvalid, xtest)
#    xtrain, ytrain, xvalid, yvalid, xtest = readCleanData()
#    xtrain = xtrain[0:30000]
#    ytrain = ytrain[0:30000]
    xtrain, xvalid, xtest, countVect = vectorizeWords(xtrain, xvalid, xtest)
    
    ytest, w, b = resumeLogisticRegression(xtrain, ytrain, xvalid, yvalid, xtest, 1)
    #1000 epochs = 30 min
    
    ytest = pd.Series(ytest, index=xtest.iloc[:,1])
    ytest.to_csv('result.csv')
    np.savetxt('w.csv', w)
    np.savetxt('b.csv', b)
    return ytest

def test():
    xtr, ytr, xte = readCleanData()
    xtr, xte, cv = vectorizeWords(xtr, xte)
    
    xtr = xtr.drop(['tid', 'date', 'id', xtr.columns[0]], axis=1)
    xte = xte.drop(['tid', 'date', 'id', xte.columns[0]], axis=1)
    x = T.matrix('x')
    y = T.ivector('y')
    w = theano.shared(np.zeros((xtr.columns.size, 9), dtype=theano.config.floatX), name='w')
    b = theano.shared(np.zeros(9, dtype=theano.config.floatX), name='b')
    
    return xtr, ytr, xte, x, y, w, b
    

if __name__=='main':
    main()
