import rmp
import pandas as pd
import numpy as np
def main():
    #TODO: normalize columns by z-score! 
#    xtrain, ytrain, xvalid, yvalid, xtest = readData()
#    xtrain, ytrain, xvalid, yvalid, xtest = cleanData(xtrain, ytrain, xvalid, yvalid, xtest)
    xtrain, ytrain, xvalid, yvalid, xtest = rmp.readCleanData(keepQuality=True)
#    xtrain = xtrain[0:30000]
#    ytrain = ytrain[0:30000]
    xtrain, xvalid, xtest, countVect = rmp.vectorizeWords(xtrain, xvalid, xtest)
    xtrain = xtrain.append(xvalid, ignore_index=True)
    xtrain = xtrain.drop(['tid', 'date', 'id', xtrain.columns[0]], axis=1)
    
    xtrain = xtrain.apply(lambda x: (x-np.mean(x))/np.std(x), axis = 0, raw=True)
    
    helpfulness = xtrain['helpfulness'].get_values()
    clarity = xtrain['clarity'].get_values()[:,0]
    easiness = xtrain['easiness'].get_values()
    quality = xtrain['quality'].get_values()[:,0]
    
    xtrain = xtrain.drop(['helpfulness', 'clarity', 'easiness', 'quality'], axis=1)
    n = len(xtrain.iloc[0,:])
    helpfulnessCorr = []
    
    for i in range(n):
        helpfulnessCorr.append(np.correlate(xtrain.iloc[:,i].get_values(), helpfulness)[0])
    clarityCorr = []
    for i in range(n):
        clarityCorr.append(np.correlate(xtrain.iloc[:,i].get_values(), clarity)[0])
    easinessCorr = []
    for i in range(n):
        easinessCorr.append(np.correlate(xtrain.iloc[:,i].get_values(), easiness)[0])
    qualityCorr = []
    for i in range(n):
        qualityCorr.append(np.correlate(xtrain.iloc[:,i].get_values(), quality)[0])
    
    correlationData = pd.DataFrame({'column': xtrain.columns, 'helpfulnessCorr': helpfulnessCorr, 'clarityCorr': clarityCorr, 'easinessCorr': easinessCorr, 'qualityCorr': qualityCorr}, columns=['column', 'helpfulnessCorr', 'clarityCorr', 'easinessCorr', 'qualityCorr'])
    
    correlationData.to_csv('correlationData.csv')
