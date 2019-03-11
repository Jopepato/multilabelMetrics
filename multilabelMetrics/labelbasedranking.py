from .auxiliar_functions import relevantIndexes, irrelevantIndexes
import numpy as np
from decimal import Decimal
def aucMacro(y_test, probabilities):
    """
    AUC Macro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Matrix of the probabilities associated to each label
    Returns
    =======
    aucMacro : float
        AUC Macro
    """
    aucmacro = 0.0
    z = np.zeros(y_test.shape[0], dtype = int)
    bar_z = np.zeros(y_test.shape[0], dtype = int)
    for i in range(y_test.shape[1]):
        withi = 0
        without = 0
        #List if instances with y and withouy y as relevant labels
        for j in range(0, int(y_test.shape[0])):
            relevantVector = relevantIndexes(y_test[j,:])
            if i in relevantVector:
                z[withi] = j
                withi += 1
            else:
                bar_z[without] = j
                without += 1
        
        auc_i = 0.0
        for j in range(0,withi):
            for k in range(0,without):
                if probabilities[int(z[j]),i] >= probabilities[int(bar_z[k]),i]:
                    auc_i += 1.0
        if withi*without !=0:
            aucmacro = aucmacro + auc_i/(withi*without)

    aucmacro = Decimal(aucmacro)/Decimal(y_test.shape[1])
    return aucmacro

def aucMicro(y_test, probabilities):
    """
    AUC Micro of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Matrix of the probabilities associated to each label
    Returns
    =======
    aucMicro : float
        AUC Micro
    """
    aucmicro = 0.0
    rel_x_irel = 0.0
    lista = np.zeros(y_test.shape[1], dtype=int)
    for i in range(y_test.shape[0]):
        relevantVector = relevantIndexes(y_test[i,:])
        rel_x_irel += len(relevantVector)
    
    rel_x_irel = rel_x_irel * (y_test.shape[0]*y_test.shape[1] - rel_x_irel)
    for i in range(y_test.shape[0]):

        for j in range(y_test.shape[0]):
            #Irrelevants labels of j-th
            a = 0
            b = 0
            c = 0
            jnext = 0
            relevantVector = relevantIndexes(y_test[j,:])
            ii = 0
            while(ii < len(relevantVector)):
                if(relevantVector[ii] == jnext):
                    ii +=1
                    jnext +=1
                else:
                    lista[c] = jnext
                    jnext += 1
                    c += 1
            
            while(jnext < int(y_test.shape[1])):
                lista[c] = jnext
                jnext +=1
                c += 1
            
            #Relevant labels for i-th
            if int(len(relevantVector)) != int(0):
                for a in range(int(len(relevantVector))):
                    for b in range(int(c)):
                        if probabilities[i, relevantVector[a]] >= probabilities[j,int(lista[b])]:
                            aucmicro += 1.0

    aucmicro = Decimal(aucmicro)/Decimal(rel_x_irel)


    return aucmicro

def aucInstance(y_test, probabilities):
    """
    AUC Instance of our model

    Params
    ======
    y_test : sparse or dense matrix (n_samples, n_labels)
        Matrix of labels used in the test phase
    probabilities: sparse or dense matrix (n_samples, n_labels)
        Matrix of the probabilities associated to each label
    Returns
    =======
    aucInstance : float
        AUC Instance
    """
    aucinstance = 0.0
    lista = np.zeros(y_test.shape[1], dtype = int)

    for i in range(y_test.shape[0]):
        suma = 0.0
        #Irrelevant labels of i-th
        c = 0
        nexti = 0
        relevantVector = relevantIndexes(y_test[i,:])
        ii = 0
        while(ii < len(relevantVector)):
            if(relevantVector[ii] == nexti):
                ii += 1
                nexti += 1
            else:
                lista[c] = nexti
                nexti += 1
                c += 1
        
        while(nexti < y_test.shape[1]):
            lista[c] = nexti
            nexti += 1
            c +=1
        #Relevant labels of i-th
        for a in range(len(relevantVector)):
            for b in range(c):
                if probabilities[i, relevantVector[a]] >= probabilities[i,int(lista[b])]:
                    suma +=1
        
        if len(relevantVector)*(y_test.shape[1]-len(relevantVector)) != 0:
            aucinstance = aucinstance + suma/(len(relevantVector)*(y_test.shape[1]*len(relevantVector)))

    aucinstance = Decimal(aucinstance)/Decimal(y_test.shape[0])
    return aucinstance