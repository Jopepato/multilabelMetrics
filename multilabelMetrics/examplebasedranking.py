import numpy as np
from functions import rankingMatrix, relevantIndexes, irrelevantIndexes
def oneError(y_test, probabilities):
    """
    One Error 

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Probability of being into a class or not per each label
    Returns
    =======
    oneError : float
        One Error
    """
    oneerror = 0.0
    ranking = rankingMatrix(probabilities)

    for i in range(y_test.shape[0]):
        index = np.argmin(ranking[i,:])
        if y_test[i,index] == 0:
            oneerror += 1.0
    
    oneerror = float(oneerror)/float(y_test.shape[0])

    return oneerror

def coverage(y_test, probabilities):
    """
    Coverage

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Probability of being into a class or not per each label
    Returns
    =======
    coverage : float
        coverage
    """
    coverage = 0.0
    ranking = rankingMatrix(probabilities)

    for i in range(y_test.shape[0]):
        coverageMax = 0.0
        for j in range(y_test.shape[1]):
            if y_test[i,j] == 1:
                if ranking[i,j] > coverageMax:
                    coverageMax = ranking[i,j]
        
        coverage += coverageMax

    coverage = float(coverage)/float(y_test.shape[0])
    coverage -= 1.0

    return coverage

def averagePrecision(y_test, probabilities):
    """
    Average Precision

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Probability of being into a class or not per each label
    Returns
    =======
    averageprecision : float
        Average Precision
    """
    averageprecision = 0.0
    averageprecisionsummatory = 0.0
    ranking = rankingMatrix(probabilities)
    
    for i in range(y_test.shape[0]):
        relevantVector =relevantIndexes(y_test, i)
        for j in range(y_test.shape[1]):
            average = 0.0
            if y_test[i, j] == 1:
                for k in range(y_test.shape[1]):
                    if(y_test[i,k] == 1):
                        if ranking[i,k] <= ranking[i,j]:
                            average += 1.0
            if ranking[i,j] != 0:
                averageprecisionsummatory += average/ranking[i,j]
        
        if len(relevantVector) == 0:
            averageprecision += 1.0
        else:
            averageprecision += averageprecisionsummatory/float(len(relevantVector))
        averageprecisionsummatory = 0.0
    
    averageprecision /= y_test.shape[0]
    return averageprecision

def rankingLoss(y_test, probabilities):
    """
    Ranking Loss

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Probability of being into a class or not per each label
    Returns
    =======
    rankingloss : float
        Ranking Loss
    """
    rankingloss = 0.0

    for i in range(y_test.shape[0]):
        relevantVector = relevantIndexes(y_test, i)
        irrelevantVector = irrelevantIndexes(y_test, i)
        loss = 0.0

        for j in range(y_test.shape[1]):
            if y_test[i,j] == 1:
                for k in range(y_test.shape[1]):
                    if y_test[i,k] == 0:
                        if float(probabilities[i,j]) <= float(probabilities[i,k]):
                            loss += 1.0
        if len(relevantVector) != 0 and len(irrelevantVector) != 0:
            rankingloss += loss/float(len(relevantVector)*len(irrelevantVector))
    
    rankingloss /= y_test.shape[0]

    return rankingloss