from decimal import Decimal
from .auxiliar_functions import HammingDistanceListOfIntegers, intersectionCardinality, unionCardinality
import numpy as np
def subsetAccuracy(y_test, y_pred):
    """
    The subset accuracy evaluates the fraction of correctly classified examples

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    subsetaccuracy : float
        Subset Accuracy of our model
    """
    subsetaccuracy = 0.0

    for i in range(y_test.shape[0]):
        same = True
        for j in range(y_test.shape[1]):
            if y_test[i,j] != y_pred[i,j]:
                same = False
                break
        if same:
            subsetaccuracy += 1.0
    
    return Decimal(subsetaccuracy)/Decimal(y_test.shape[0])


def hammingLoss(y_test, y_pred):
    """
    The hamming loss evaluates the fraction of misclassified instance-label pairs

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    hammingloss : float
        Hamming Loss of our model
    """
    hammingloss = 0.0
    for i in range(y_test.shape[0]):
        hammingDistance = HammingDistanceListOfIntegers(y_test[i,:], y_pred[i,:].A1)
        hammingloss = Decimal(hammingloss) + Decimal(hammingDistance)/Decimal(y_test.shape[1])
    
    return Decimal(hammingloss)/Decimal(y_test.shape[0])

def eb_accuracy(y_test, y_pred):
    """
    Example based accuracy of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracy : float
        Accuracy of our model
    """
    accuracy = 0.0
    intersectionArray = intersectionCardinality(y_test, y_pred)
    unionArray = unionCardinality(y_test, y_pred)
    
    for i in range(y_test.shape[0]):
        if unionArray[i] != 0:
            accuracy += intersectionArray[i]/float(unionArray[i])

    accuracy = Decimal(accuracy)/Decimal(y_test.shape[0])

    return accuracy



def eb_precision(y_test, y_pred):
    """
    Example based precision of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precision : float
        Precision of our model
    """
    precision = 0.0

    intersectionArray = intersectionCardinality(y_test, y_pred)
    for i in range(y_test.shape[0]):
        if np.sum(y_pred[i,:]) != 0:
            precision = precision + float(intersectionArray[i])/float(np.sum(y_pred[i,:]))
            
    return Decimal(precision)/Decimal(y_test.shape[0])


def eb_recall(y_test, y_pred):
    """
    Example based recall of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recall : float
        recall of our model
    """
    recall = 0.0

    interesectionArray = intersectionCardinality(y_test,y_pred)

    for i in range(y_test.shape[0]):
        if sum(np.array(y_test[i,:])):
            recall += float(interesectionArray[i])/float(sum(np.array(y_test[i,:])))

    return Decimal(recall)/Decimal(y_test.shape[0])



def eb_fbeta(y_test, y_pred, beta=1):
    """
    Example based FBeta of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbeta : float
        fbeta of our model
    """
    pr = eb_precision(y_test, y_pred)
    re = eb_recall(y_test, y_pred)

    num = float((1+pow(beta,2))*pr*re)
    den = float(pow(beta,2)*pr + re)

    if den != 0:
        fbeta = Decimal(num)/Decimal(den)
    else:
        fbeta = 0.0
    return fbeta
