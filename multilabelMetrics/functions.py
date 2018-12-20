#Auxiliary functions
import numpy as np
def relevantIndexes(matrix, row):
    """
    Gets the relevant indexes of a vector
    """
    relevant = []
    for j in range(matrix.shape[1]):
        if matrix[row,j] == 1:
            relevant.append(int(j))
    
    return relevant


def irrelevantIndexes(matrix, row):
    """
    Gets the irrelevant indexes of a vector
    """
    irrelevant = []
    for j in range(matrix.shape[1]):
        if matrix[row,j] == 0:
            irrelevant.append(int(j))
    
    return irrelevant

def multilabelConfussionMatrix(y_test, predictions):
    """
    Returns the TP, FP, TN, FN
    """
    TP = np.zeros(y_test.shape[1])
    FP = np.zeros(y_test.shape[1])
    TN = np.zeros(y_test.shape[1])
    FN = np.zeros(y_test.shape[1])

    for j in range(y_test.shape[1]):
        TPaux = 0
        FPaux = 0
        TNaux = 0
        FNaux = 0
        for i in range(y_test.shape[0]):
            if int(y_test[i,j]) == 1:
                if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                    TPaux += 1
                else:
                    FPaux += 1
            else:
                if int(y_test[i,j]) == 0 and int(predictions[i,j]) == 0:
                    TNaux += 1
                else:
                    FNaux += 1
        TP[j] = TPaux
        FP[j] = FPaux
        TN[j] = TNaux
        FN[j] = FNaux

    return TP, FP, TN, FN

def multilabelMicroConfussionMatrix(TP, FP, TN, FN):
    TPMicro = 0.0
    FPMicro = 0.0
    TNMicro = 0.0
    FNMicro = 0.0
    
    for i in range(len(TP)):
        TPMicro = TPMicro + TP[i]
        FPMicro = FPMicro + FP[i]
        TNMicro = TNMicro + TN[i]
        FNMicro = FNMicro + FN[i]
    
    return TPMicro, FPMicro, TNMicro, FNMicro

def rankingMatrix(probabilities):
    """
    Matrix with the rankings for each label
    """
    ranking = np.zeros(shape=[probabilities.shape[0], probabilities.shape[1]])
    probCopy = np.copy(probabilities)
    for i in range(probabilities.shape[0]):
        indexMost = 0
        iteration = 1
        while(sum(probCopy[i,:]) != 0):
            for j in range(probabilities.shape[1]):
                if probCopy[i,j] > probCopy[i,indexMost]:
                    indexMost = j
                ranking[i, indexMost] = iteration
                probCopy[i, indexMost] = 0
                iteration += 1
    
    return ranking