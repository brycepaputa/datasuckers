import basicLogisticRegression
import rmp
import numpy as np
import pandas as pd

#    xtrain, ytrain, xvalid, yvalid, xtest = readData()
xtrain, ytrain, xvalid, yvalid, xtest = rmp.readData()
xtrain, ytrain, xvalid, yvalid, xtest = rmp.cleanData(xtrain, ytrain, xvalid, yvalid, xtest, keepQuality=True)
#    xtrain = xtrain[0:30000]
#    ytrain = ytrain[0:30000]
xtrain, xvalid, xtest, countVect = rmp.vectorizeWords(xtrain, xvalid, xtest)

yvalid, ytest, w, b = basicLogisticRegression.resumeLogisticRegression(xtrain, ytrain, xvalid, yvalid, xtest, 1)
#1000 epochs = 30 min

ytest = pd.Series(ytest, index=xtest.iloc[:,0])
yvalid = pd.Series(yvalid, index=xvalid.iloc[:,0])
error = yvalid.get_values()-xvalid.quality.get_values()
errorWRTHC = np.zeros((5, 5))
countWRTHC = np.zeros((5, 5))
for i in range(len(xvalid)):
    errorWRTHC[xvalid.iloc[i].helpfulness-1, xvalid.iloc[i].clarity-1] += error[i]
    countWRTHC[xvalid.iloc[i].helpfulness-1, xvalid.iloc[i].clarity-1] += 1
