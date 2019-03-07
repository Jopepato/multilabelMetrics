def readParams (paramsFileName):
    "This functions reads the params from a file and store them in a dictionary"
    file = open(paramsFileName, 'r')
    
    paramNames = []
    paramAttributes = []
    #We are going to create a dictionary from both lists
    for line in file.readlines():
        data = line.split(' : ')
        paramNames.append(str(data[0]))
        aux = data[1]
        paramAttributes.append(aux[0:len(aux)-1])

    paramDictionary = {}

    file.close()

    for i in range(len(paramNames)):
        
        paramDictionary[paramNames[i]] = paramAttributes[i]

    del paramNames
    del paramAttributes
    return paramDictionary

def readDataFromFile (fileName):
    "This functions reads data from a file and store it in two matrices"
    #Open the file
    file = open(fileName, 'r')

    #Now we have to read the first line and check if it's sparse or dense
    firstLine = file.readline()
    words = firstLine.split()
    word = words[1]
    if word[:-1] == 'SPARSE':
        sparse = True #The file is in sparse mode
    else:
        sparse = False #The file is in dense mode


    secondLine = file.readline()
    words = secondLine.split()
    instances = int(words[1])
    thirdLine = file.readline()
    words = thirdLine.split()
    attributes = int(words[1])
    fourthLine = file.readline()
    words = fourthLine.split()
    labels = int(words[1])
    #Now we do a loop reading all the other lines
    #Then we read the file, different way depending if sparse or dense

    #The loop starts in the first line of data
    #We have to store that data in two matrices
    X = np.zeros((instances, attributes), dtype=float)
    y = np.zeros((instances, labels), dtype=int)
    numberLine = 0
    for line in file.readlines():
        putToX = True
        firstIndex = 1
        numberData = 0
        numberY = 0
        for data in line.split():
            if sparse:#Sparse format, we have to split each data 
                if data == '[':
                    putToX = False

                if putToX == True and (data != '[' and data != ']'):
                    sparseArray = data.split(':')
                    lastIndex = int(sparseArray[0])
                    for i in range(firstIndex, lastIndex - 1):
                        X[numberLine, i-1] = float(0)
                    X[numberLine, lastIndex-1] = float(sparseArray[1])
                    firstIndex = lastIndex-1
                else:
                    if (data != '[') and (data != ']'):
                        aux = float(data)
                        y[numberLine, numberY] = int(aux)
                        numberY += 1
                
            else:#Dense format
                if data == '[':
                    putToX = False

                if putToX == True and (data != '[' and data != ']'):
                    X[numberLine, numberData] = float(data)
                else:
                    if (data != '[') and (data != ']'):
                        #This is good for the dense format
                        aux = float(data)
                        y[numberLine, numberY] = int(aux)
                        numberY += 1
            numberData += 1
        
        numberLine += 1
    X = sp.csr_matrix(X)
    file.close()
    return X, y