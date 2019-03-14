#Auxiliary functions
import numpy as np
from decimal import Decimal
def relevantIndexes(vector):
    """
    Gets the relevant indexes of a vector
    """
    relevant = []
    for i in range(len(vector)):
        if vector[i] == 1:
            relevant.append(int(i))
    
    return relevant


def irrelevantIndexes(vector):
    """
    Gets the irrelevant indexes of a vector
    """
    irrelevant = []
    for i in range(len(vector)):
        if vector[i] == 0:
            irrelevant.append(int(i))
    
    return irrelevant

def multilabelConfussionMatrix(y_test, y_pred):
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
                if int(y_test[i,j]) == 1 and int(y_pred[i,j]) == 1:
                    TPaux += 1
                else:
                    FPaux += 1
            else:
                if int(y_test[i,j]) == 0 and int(y_pred[i,j]) == 0:
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
    
    TPMicro = Decimal(TPMicro)/Decimal(len(TP))
    FPMicro = Decimal(FPMicro)/Decimal(len(TP))
    TNMicro = Decimal(TNMicro)/Decimal(len(TP))
    FNMicro = Decimal(FNMicro)/Decimal(len(TP))

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
                if float(probCopy[i,j]) > float(probCopy[i,indexMost]):
                    indexMost = j
                ranking[i, indexMost] = iteration
                probCopy[i, indexMost] = 0
                iteration += 1
    
    return ranking

def intersectionCardinality(y_test, y_pred):
    interesectionArray = np.zeros(y_test.shape[0],dtype=int)
    for i in range(y_test.shape[0]):
        intersection = 0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) == 1 and int(y_pred[i,j] == 1):
                intersection += 1
        interesectionArray[i] = intersection

    return interesectionArray

def unionCardinality(y_test, y_pred):
    unionArray = np.zeros(y_test.shape[0], dtype=int)
    for i in range(y_test.shape[0]):
        union  = 0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j] == 1) or int(y_pred[i,j] == 1):
                union += 1
        
        unionArray[i] = union

    return unionArray

def HammingDistanceListOfIntegers(y_true, y_pred):
    """
    Returns the hamming distance
    """
    hamming = 0.0
    x_index = 0
    y_index  = 0
    x_length = float(np.sum(y_true))
    y_length = float(np.sum(y_pred))

    if (x_length ==0 or y_length==0):
        hamming = x_length + y_length
    else:
        while(x_index < x_length and y_index < y_length):
            if(y_true[x_index] == y_pred[y_index]):
                x_index += 1
                y_index +=1
            else:
                hamming += 1
                if y_true[x_index] < y_pred[y_index]:
                    x_index += 1
                else:
                    y_index += 1
        hamming += x_length-x_index + y_length-y_index
    
    return hamming