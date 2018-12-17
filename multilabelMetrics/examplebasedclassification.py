def subsetAccuracy(y_test, predictions):
    """
    The subset accuracy evaluates the fraction of correctly classified examples

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
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
            if y_test[i,j] != predictions[i,j]:
                same = False
                break
        if same:
            subsetaccuracy += 1.0
    
    return subsetaccuracy/y_test.shape[0]


def hammingLoss(y_test, predictions):
    """
    The hamming loss evaluates the fraction of misclassified instance-label pairs

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    hammingloss : float
        Hamming Loss of our model
    """
    hammingloss = 0.0
    for i in range(y_test.shape[0]):
        aux = 0.0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) != int(predictions[i,j]):
                aux = aux+1.0
        aux = aux/y_test.shape[1]
        hammingloss = hammingloss + aux
    
    return hammingloss/y_test.shape[0]

def accuracy(y_test, predictions):
    """
    Accuracy of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    accuracy : float
        Accuracy of our model
    """
    accuracy = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        union = 0.0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) == 1 or int(predictions[i,j]) == 1:
                union += 1
            if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                intersection += 1
            
        if union != 0:
            accuracy = accuracy + float(intersection/union)

    accuracy = float(accuracy/y_test.shape[0])

    return accuracy



def precision(y_test, predictions):
    """
    Precision of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    precision : float
        Precision of our model
    """
    precision = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        hXi = 0.0
        for j in range(y_test.shape[1]):
            hXi = hXi + int(predictions[i,j])
            if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                intersection += 1
            
        if hXi != 0:
            precision = precision + float(intersection/hXi)
            

    precision = float(precision/y_test.shape[0])
    
    return precision


def recall(y_test, predictions):
    """
    Recall of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    recall : float
        recall of our model
    """
    recall = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        Yi = 0.0
        for j in range(y_test.shape[1]):
            Yi = Yi + int(y_test[i,j])

            if y_test[i,j] == 1 and int(predictions[i,j]) == 1:
                intersection = intersection + 1
    
        if Yi != 0:
            recall = recall + float(intersection/Yi)    
    
    recall = recall/y_test.shape[0]
    return recall



def fbeta(y_test, predictions, beta=1):
    """
    FBeta of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    predictions: sparse or dense matrix (n_samples, n_labels)
        Matrix of predicted labels given by our model
    Returns
    =======
    fbeta : float
        fbeta of our model
    """
    pr = precision(y_test, predictions)
    re = recall(y_test, predictions)

    num = float((1+pow(beta,2))*pr*re)
    den = float(pow(beta,2)*pr + re)

    if den != 0:
        fbeta = num/den
    else:
        fbeta = 0.0
    return fbeta
