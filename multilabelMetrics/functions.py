#Auxiliary functions

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