from .functions import multilabelConfussionMatrix, multilabelMicroConfussionMatrix
def accuracyMacro(y_test, predictions):
    """
    Accuracy Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracymacro : float
        Accuracy Macro of our model
    """
    accuracymacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        accuracymacro = accuracymacro + ((TP[i] + TN[i])/(TP[i] + FP[i] + TN[i] + FN[i]))
    
    accuracymacro = float(accuracymacro/len(TP))

    return accuracymacro


def accuracyMicro(y_test, predictions):
    """
    Accuracy Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracymicro : float
        Accuracy Micro of our model
    """
    accuracymicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FPMicro + TNMicro + FNMicro) != 0:
        accuracymicro = float((TPMicro+TNMicro)/(TPMicro + FPMicro + TNMicro + FNMicro))

    return accuracymicro


def precisionMacro(y_test, predictions):
    """
    Precision Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precisionmacro : float
        Precision macro of our model
    """
    precisionmacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        if TP[i] + FP[i] != 0:
            precisionmacro = precisionmacro + (TP[i]/(TP[i] + FP[i]))

    precisionmacro = float(precisionmacro/len(TP))
    return precisionmacro


def precisionMicro(y_test, predictions):
    """
    Precision Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precisionmicro : float
        Precision micro of our model
    """
    precisionmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)
    if (TPMicro + FPMicro) != 0:
        precisionmicro = float(TPMicro/(TPMicro + FPMicro))


    return precisionmicro

def recallMacro(y_test, predictions):
    """
    Recall Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recallmacro : float
        Recall Macro of our model
    """
    recallmacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        if TP[i] + FN[i] != 0:
            recallmacro = recallmacro + (TP[i]/(TP[i] + FN[i]))

    recallmacro = recallmacro/len(TP)
    return recallmacro

def recallMicro(y_test, predictions):
    """
    Recall Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recallmicro : float
        Recall Micro of our model
    """
    recallmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FNMicro) != 0:
        recallmicro = float(TPMicro/(TPMicro + FNMicro))

    return recallmicro


def fbetaMacro(y_test, predictions, beta=1):
    """
    FBeta Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbetamacro : float
        FBeta Macro of our model
    """
    fbetamacro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    
    for i in range(len(TP)):
        num = float((1+pow(beta,2))*TP[i])
        den = float((1+pow(beta,2))*TP[i] + pow(beta,2)*FN[i] + FP[i])
        if den != 0:
            fbetamacro = fbetamacro + num/den

    fbetamacro = fbetamacro/len(TP)
    return fbetamacro

def fbetaMicro(y_test, predictions, beta=1):
    """
    FBeta Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbetamicro : float
        FBeta Micro of our model
    """
    fbetamicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    num = float((1+pow(beta,2))*TPMicro)
    den = float((1+pow(beta,2))*TPMicro + pow(beta,2)*FNMicro + FPMicro)
    fbetamicro = float(num/den)

    return fbetamicro