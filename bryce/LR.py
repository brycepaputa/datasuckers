import rmp
import cPickle
import theano
import theano.tensor as T
import numpy as np
import timeit

class LogisticRegression:
    def __init__(self, x, nIn, nOut):
        self.w = theano.shared(
            value = np.zeros((nIn, nOut), dtype = theano.config.floatX),
            name = 'W',
            borrow = True
        )
        self.b = theano.shared(
            value = np.zeros((nOut,), dtype = theano.config.floatX),
            name = 'b',
            borrow = True
        )
        self.P_y = T.nnet.softmax(T.dot(x, self.w) + self.b)
        
        self.yHat = T.argmax(self.P_y, axis = 1)
        
        self.params = [self.w, self.b]
        
        self.x = x

    def categorical_crossentropy(self, y):
        return T.nnet.categorical_crossentropy(self.P_y, y)
        
    def regularization(self, factor):
        return factor * (self.w**2).mean()
        
    def error(self, y):
        if y.ndim != self.yHat.ndim:
            raise TypeError('y should have the same shape as yHat', ('y', y.type, 'yHat', self.yHat.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(y, self.yHat))
        else:
            raise NotImplementedError()
            
def loadRMPData(useCleaned = False):
    if useCleaned:       
        xtrain, ytrain, xvalid, yvalid, xtest = rmp.readCleanData()
    else:
        xtrain, ytrain, xvalid, yvalid, xtest = readData()
        xtrain, ytrain, xvalid, yvalid, xtest = cleanData(xtrain, ytrain, xvalid, yvalid, xtest)        
    xtrain, xvalid, xtest, countVect = rmp.vectorizeWords(xtrain, xvalid, xtest)
    xtrain = xtrain.drop(['tid', 'date', 'id', xtrain.columns[0]], axis=1)
    xvalid = xvalid.drop(['tid', 'date', 'id', xvalid.columns[0]], axis=1)
    xtest = xtest.drop(['tid', 'date', 'id', xtest.columns[0]], axis=1)
    ytest = np.zeros((len(xtest),), dtype = theano.config.floatX)
    
    def shareData(dataX, dataY, borrow=True):
        sharedX = theano.shared(np.asarray(dataX, dtype = theano.config.floatX), borrow=True)
        sharedY = theano.shared(np.asarray(dataY, dtype = theano.config.floatX), borrow=True)
        return sharedX, T.cast(sharedY, 'int32')
    
    xtrain, ytrain = shareData(xtrain, ytrain)
    xvalid, yvalid = shareData(xvalid, yvalid)
    xtest, ytest = shareData(xtest, ytest)
    return [(xtrain, ytrain), (xvalid, yvalid), (xtest, ytest)]
    
def RMPSGD(learningRate=.13, epochs=1000, useCleanedData=True, batchSize=600, regularization=.001):
    datasets = loadRMPData(useCleanedData)
    
    xtrain, ytrain = datasets[0]
    xvalid, yvalid = datasets[1]
    xtest, ytest = datasets[2]
    
    trainBatches = xtrain.get_value(borrow=True).shape[0] / batchSize
    validBatches = xvalid.get_value(borrow=True).shape[0] / batchSize
    testBatches = xtest.get_value(borrow=True).shape[0] / batchSize
    
    print '... building the model'
    
    index = T.lscalar()
    
    x = T.matrix('x')
    y = T.ivector('y')
    
    classifier = LogisticRegression(x, xtrain.get_value(borrow=True).shape[1], 9)
    
    cost = classifier.categorical_crossentropy(y) + classifier.regularization(regularization)
    
    validateModel = theano.function(
        inputs=[index],
        outputs=classifier.error(y),
        givens={
            x: xvalid[index * batchSize : (index+1) * batchSize],
            y: yvalid[index * batchSize : (index+1) * batchSize]
        }
    )
    
    gw, gb = T.grad(T.mean(cost), [classifier.w, classifier.b])
    
    trainUpdates = [(classifier.w, classifier.w - learningRate * gw), (classifier.b, classifier.b - learningRate * gb)]
    
    trainModel = theano.function(
        inputs = [index],
        outputs = cost,
        updates = trainUpdates,
        givens = {
            x: xtrain[index * batchSize : (index+1) * batchSize],
            y: ytrain[index * batchSize : (index+1) * batchSize]
        }
    )
    
    print '... training the model'
    patience = 5000 #minimum samples to look at (minibatches)
    patienceInc = 2 #look at this many more whenever a new best is found
    improveThresh = .995 #how much is considered an improvement
    validFreq = min(trainBatches, patience / 2) #how many minibatches to go through before validation
    bestValidLoss = np.inf
    
    startTime = timeit.default_timer()
    
    done = False
    epoch = 0
    while epoch < epochs and not done:
        epoch = epoch + 1
        for minibatch in xrange(trainBatches):
            minibatchCost = trainModel(minibatch)
            iter = (epoch-1)*trainBatches+minibatch
            if (iter+1) % validFreq == 0:
                validLoss = [validateModel(i) for i in xrange(validBatches)]
                validLoss = np.mean(validLoss)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                    (epoch, minibatch + 1, trainBatches, validLoss*100)
                )
                if validLoss < bestValidLoss:
                    if validLoss < bestValidLoss * improveThresh:
                        patience = max(patience, iter * patienceInc)
                    bestValidLoss = validLoss
                    with open('bestLR.pkl', 'w') as f:
                        cPickle.dump(classifier, f)
            if patience <= iter:
                done = True
                break    
                
    endTime = timeit.default_timer()
    print ('Optimization complete with best validation score of %f %%' % (bestValidLoss*100))             
                
    
def predict():
    
    classifier = cPickle.load(open('bestModel.pkl'))
    
    predict = theano.function(inputs = [classifier.input], outputs = classifier.yHat)
    
    datasets = loadRMPData(False)
    xtest, ytest = datasets[2]
    predSample = predict(xtest[:10])
    print "Predicted values for the first 10 examples in test set:"
    print predSample
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
