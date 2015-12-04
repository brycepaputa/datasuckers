import rmp
import numpy as np
import pandas as pd
import theano
from theano import tensor as T

def resumeLogisticRegression(xtrain, ytrain, xvalid, yvalid, xtest, numSteps, trainingRate = .1, alpha = .001):
    w = theano.shared(np.loadtxt('w.csv', dtype=theano.config.floatX), name='w', borrow=True)
    b = theano.shared(np.loadtxt('b.csv', dtype=theano.config.floatX), name='b', borrow=True)
    return logisticRegression(xtrain, ytrain, xvalid, yvalid, xtest, numSteps, trainingRate, alpha, w, b)


def logisticRegression(xtrain, ytrain, xvalid, yvalid, xtest, numSteps, trainingRate = .1, alpha = .001, w = None, b = None):
    xtrain = xtrain.drop(['tid', 'date', 'id', xtrain.columns[0]], axis=1)
    xvalid = xvalid.drop(['tid', 'date', 'id', xvalid.columns[0]], axis=1)
    xtest = xtest.drop(['tid', 'date', 'id', xtest.columns[0]], axis=1)
    if 'quality' in xtrain.columns:
        xtrain = xtrain.drop(['helpfulness', 'clarity', 'easiness', 'quality'], axis=1)
        xvalid = xvalid.drop(['helpfulness', 'clarity', 'easiness', 'quality'], axis=1)
    x = T.matrix('x')
    y = T.ivector('y')
    if w == None or b == None:
        w = theano.shared(np.zeros((xtrain.columns.size, 9), dtype=theano.config.floatX), name='w', borrow=True)
        b = theano.shared(np.zeros(9, dtype=theano.config.floatX), name='b', borrow=True)

    P_y = T.nnet.softmax(T.dot(x, w)+b)
    prediction = T.argmax(P_y, axis=1)
    xent = T.mean(T.nnet.categorical_crossentropy(P_y, y))
    cost = xent + alpha * (w**2).mean()
    gw, gb = T.grad(cost, [w, b])
    
    validationError = T.mean((prediction - y) ** 2)
    
    train = theano.function(inputs = [x, y],
                            outputs = [],
                            updates = ((w, w - trainingRate*gw), (b, b - trainingRate*gb)))
    validate = theano.function(inputs = [x, y], outputs = validationError)
    predict = theano.function(inputs = [x], outputs = prediction)
    
    for i in range(numSteps):
        train(xtrain, ytrain)
        print i
        print "\t" + str(validate(xvalid, yvalid))
    
    ytest = predict(xtest)+2
    yvalid = predict(xvalid)+2
    return yvalid, ytest, w.get_value(), b.get_value()
    
    
def main():
#    xtrain, ytrain, xvalid, yvalid, xtest = readData()
#    xtrain, ytrain, xvalid, yvalid, xtest = cleanData(xtrain, ytrain, xvalid, yvalid, xtest)
    xtrain, ytrain, xvalid, yvalid, xtest = rmp.readCleanData()
#    xtrain = xtrain[0:30000]
#    ytrain = ytrain[0:30000]
    xtrain, xvalid, xtest, countVect = rmp.vectorizeWords(xtrain, xvalid, xtest)
    
    ytest, w, b = resumeLogisticRegression(xtrain, ytrain, xvalid, yvalid, xtest, 1)
    #1000 epochs = 30 min
    
    ytest = pd.Series(ytest, index=xtest.iloc[:,1])
    ytest.to_csv('result.csv')
    np.savetxt('w.csv', w)
    np.savetxt('b.csv', b)
    return xtrain, ytrain, xvalid, yvalid, xtest, ytest

#def test():
#    xtr, ytr, xte = readCleanData()
#    xtr, xte, cv = vectorizeWords(xtr, xte)
#    
#    xtr = xtr.drop(['tid', 'date', 'id', xtr.columns[0]], axis=1)
#    xte = xte.drop(['tid', 'date', 'id', xte.columns[0]], axis=1)
#    x = T.matrix('x')
#    y = T.ivector('y')
#    w = theano.shared(np.zeros((xtr.columns.size, 9), dtype=theano.config.floatX), name='w')
#    b = theano.shared(np.zeros(9, dtype=theano.config.floatX), name='b')
#    
#    return xtr, ytr, xte, x, y, w, b
