from .auxiliar_functions import multilabelConfussionMatrix, multilabelMicroConfussionMatrix
from decimal import Decimal
def accuracyMacro(y_test, y_pred):
    """
    Accuracy Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracymacro : float
        Accuracy Macro of our model
    """
    accuracymacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, y_pred)
    for i in range(len(TP)):
        accuracymacro = Decimal(accuracymacro) + Decimal(TP[i] + TN[i])/Decimal(TP[i] + FP[i] + TN[i] + FN[i])
    
    accuracymacro = Decimal(accuracymacro)/Decimal(y_test.shape[1])

    return accuracymacro


def accuracyMicro(y_test, y_pred):
    """
    Accuracy Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracymicro : float
        Accuracy Micro of our model
    """
    accuracymicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, y_pred)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    accuracymicro = Decimal(TPMicro+TNMicro)/Decimal(y_test.shape[0])

    return accuracymicro


def precisionMacro(y_test, y_pred):
    """
    Precision Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precisionmacro : float
        Precision macro of our model
    """
    precisionmacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, y_pred)
    for i in range(len(TP)):
        if TP[i] + FP[i] != 0:
            precisionmacro = Decimal(precisionmacro) + Decimal(TP[i])/Decimal(TP[i] + FP[i])

    precisionmacro = Decimal(precisionmacro)/Decimal(y_test.shape[1])
    return precisionmacro


def precisionMicro(y_test, y_pred):
    """
    Precision Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precisionmicro : float
        Precision micro of our model
    """
    precisionmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, y_pred)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)
    if (TPMicro + FPMicro) != 0:
        precisionmicro = Decimal(TPMicro)/Decimal(TPMicro + FPMicro)


    return precisionmicro

def recallMacro(y_test, y_pred):
    """
    Recall Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recallmacro : float
        Recall Macro of our model
    """
    recallmacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, y_pred)
    for i in range(len(TP)):
        if TP[i] + FN[i] != 0:
            recallmacro = recallmacro + (TP[i]/(TP[i] + FN[i]))

    recallmacro = recallmacro/len(TP)
    return recallmacro

def recallMicro(y_test, y_pred):
    """
    Recall Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recallmicro : float
        Recall Micro of our model
    """
    recallmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, y_pred)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FNMicro) != 0:
        recallmicro = Decimal(TPMicro)/Decimal(TPMicro + FNMicro)

    return recallmicro


def fbetaMacro(y_test, y_pred, beta=1):
    """
    FBeta Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbetamacro : float
        FBeta Macro of our model
    """
    fbetamacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, y_pred)
    
    for i in range(len(TP)):
        num = float((1+pow(beta,2))*TP[i])
        den = float((1+pow(beta,2))*TP[i] + pow(beta,2)*FN[i] + FP[i])
        if den != 0:
            fbetamacro = Decimal(fbetamacro) + Decimal(num)/Decimal(den)

    fbetamacro = Decimal(fbetamacro)/Decimal(y_test.shape[1])
    return fbetamacro

def fbetaMicro(y_test, y_pred, beta=1):
    """
    FBeta Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    y_pred: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbetamicro : float
        FBeta Micro of our model
    """
    fbetamicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, y_pred)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    num = float((1+pow(beta,2))*TPMicro)
    den = float((1+pow(beta,2))*TPMicro + pow(beta,2)*FNMicro + FPMicro)
    fbetamicro = Decimal(num)/Decimal(den)

    return fbetamicro