from functions import multilabelConfussionMatrix, multilabelMicroConfussionMatrix
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


    return recallmicro


def fbetaMacro(y_test, predictions, beta):
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


    return fbetamacro

def fbetaMicro(y_test, predictions, beta):
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


    return fbetamicro