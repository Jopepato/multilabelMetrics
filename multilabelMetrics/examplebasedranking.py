import numpy as np
from .auxiliar_functions import rankingMatrix, relevantIndexes, irrelevantIndexes
from decimal import Decimal
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
        relevantVector = relevantIndexes(y_test[i,:])
        index = np.argmin(ranking[i,:])
        if int(index) not in relevantVector:
            oneerror += 1.0
    
    oneerror = Decimal(oneerror)/Decimal(y_test.shape[0])

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

    coverage = Decimal(coverage)/Decimal(y_test.shape[0])
    coverage -= 1

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
    ranking = rankingMatrix(probabilities)
    for i in range(y_test.shape[0]):
        average = 0.0
        relevantVector = relevantIndexes(y_test[i,:])
        for j in range(len(relevantVector)):
            c = 0
            fraction = 0.0
            for k in range(y_test.shape[1]):
                if(probabilities[i,k] >= probabilities[i,relevantVector[j]]):
                    c +=1
                    if int(k) in relevantVector:
                        fraction +=1

            average = average + fraction/c
        if(len(relevantVector) > 0):
            averageprecision = averageprecision + average/len(relevantVector)
    
    averageprecision = Decimal(averageprecision)/Decimal(y_test.shape[0])

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
    for i in range(0, y_test.shape[0]):
        relevantVector = relevantIndexes(y_test[i])
        irrelevantVector = irrelevantIndexes(y_test[i])
        loss = 0.0

        for j in range(len(relevantVector)):
            for k in range(len(irrelevantVector)):
                if probabilities[i,relevantVector[j]] <= probabilities[i, irrelevantVector[k]]:
                    loss +=1

        if len(relevantVector)*len(irrelevantVector) != 0:
            dim = len(relevantVector)*len(irrelevantVector)
            rankingloss = Decimal(rankingloss) + Decimal(loss)/Decimal(dim)

    rankingloss = Decimal(rankingloss)/Decimal(y_test.shape[0])

    return rankingloss