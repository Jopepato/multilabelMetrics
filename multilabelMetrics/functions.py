#Auxiliary functions
import numpy as np
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
        if vector[0] == 0:
            irrelevant.append(int(i))
    
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