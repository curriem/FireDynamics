import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")

def openFile(filePath):
    p = pickle.load(open(filePath, 'rb'))
    data = p['TS']
    #return data
    return data[85:110, :32, :32]

def modelCadence(cadence, model, tempThresh):

    fireInds = np.array(np.where(cadence > tempThresh))
    fireInds = fireInds.T
    model[fireInds[:,0],fireInds[:,1]] = 1

    
    return model

def nearestPixels(mainCoords, prevModel, d):
 
    x = mainCoords[0]
    y = mainCoords[1]

    
    nearestPixVals = prevModel[x-d:x+d+1, y-d:y+d+1]

    return nearestPixVals                                                           

def laplacian(T, coords, ninePoint=False):
    """
    this function computes the lapacian for a given matrix Z
    args:
        Z is the neighboring pixels for a given center pixel
        stencil is the stencil model to use. default is five point, 1 is nine point
    returns:
        the laplacian (a number)
    """
    x = coords[0]
    y = coords[1]
    Tc = T[x,y]
    Tt = T[x,y+1]
    Tl = T[x-1,y]
    Tr = T[x+1,y]
    Tb = T[x,y-1]
    laplacian = Tt + Tl + Tr + Tb - 4*Tc
    if ninePoint:
        Ttl = T[x-1,y+1]
        Ttr = T[x+1,y+1]
        Tbl = T[x-1,y-1]
        Tbr = T[x+1,y-1]
        laplacian = laplacian*4 + (Ttl + Ttr + Tbl + Tbr)/np.sqrt(2) - 4*Tc
    return laplacian

def firstTimeDerivative(timeseries):
    return np.diff(timeseries, axis=0)

def computeHeatFlux(neighboringPixels):
    mainPixValue = neighboringPixels[1, 1]
    heatFlux = [mainPixValue]
    i = 0 
    while i < 3:
        j = 0
        while j < 3:
            coordDiff = np.array([i,j]) - np.array([1,1])
            distance = np.sqrt(np.sum(coordDiff**2))
            if distance ==0:
                j+=1
                continue
            tempDiff = neighboringPixels[i, j] - mainPixValue
            heatFlux.append(tempDiff/distance)
            j+=1
        i+=1
            
    return sum(heatFlux)        
        
    
    
def plotCadence(cadence, title=None, minVal=0, maxVal=2):

    plt.figure()
    plt.axis('off')
    plt.title(title)
    
    plt.imshow(cadence, interpolation="nearest", vmin=minVal, vmax=maxVal)#,cmap=cmap, norm=norm)
    plt.colorbar()
    plt.show()
    plt.close()
    

def makeHist(vector, numBins):
    #makes a histogram
    plt.figure()
    plt.hist(vector, numBins)
    plt.show()
    

def plotFireFront(models, subs):
    d = 1 # layers for neighboring pixels
    
    #plotCadence(models[0], "model")
    prevModel = models[0]
    neighborList = []
    for model, sub in zip(models[1:], subs):
        frameNeighborList = []
        testSub = np.copy(sub)
        testSub *= 2
        plotCadence(prevModel+testSub)
        
        newFireInds = np.array(np.where(sub == 1))
        newFireInds = newFireInds.T
        for coords in newFireInds:
            nearestPixVals = nearestPixels(coords, prevModel, d).flatten()
            if nearestPixVals.size != 0:
                neighborList.append(np.sum(nearestPixVals))
                frameNeighborList.append(np.sum(nearestPixVals))
            
        prevModel = model
    numBins = (2*d+1)**2
    makeHist(neighborList, numBins)

def computeDiffCoeff(lap, firstTD):

    return firstTD/lap

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def main():
    plt.clf()
    plt.close()
    filePath = '/Users/mcurrie/FireDynamics/data/13fsingle3/timeseries.p'
    tempThresh = 550.
    data = openFile(filePath)
    
    # MODEL STUFF
#==============================================================================
#     models = []
#     model = np.zeros_like(data[0])
#     for cadence in data:
#         model = modelCadence(cadence, model, tempThresh)
#         modelCopy = np.copy(model)
#         models.append(modelCopy)
# 
#     models = np.array(models)
# 
#==============================================================================

    # COMPUTE DIFFUSION COEFFICIENT
    # need: first derivative in time  and laplacian
    
    t, x, y = data.shape
    n = 0
    laplacianFrames = []
    for cadence in data[1:]:
        laplacianFrame = np.zeros_like(cadence)
        i = 1
        while i < x-1:
            j=1
            while j < y-1:
                mainCoords = [i, j]
                lap = laplacian(cadence, mainCoords)
                laplacianFrame[i, j] = lap
                j+=1
            i+=1
        laplacianFrames.append(laplacianFrame[1:-1, 1:-1])

        n+=1
    firstTimeDerivativeFrames = firstTimeDerivative(data[:, 1:-1, 1:-1])
    
    #diffusionCoeffs = computeDiffCoeff(laplacianFrames, firstTimeDerivativeFrames)
    for l, f in zip(laplacianFrames, firstTimeDerivativeFrames):
        
        diffCoeff = div0(f, l)
        plotCadence(diffCoeff, minVal=np.min(diffCoeff), maxVal=np.max(diffCoeff))

    assert False 
    for d in diffusionCoeffs:
        plotCadence(diffusionCoeffs, minVal=np.min(diffusionCoeffs), maxVal=np.max(diffusionCoeffs))
    assert False
    for l, f in zip(laplacianFrames, firstTimeDerivativeFrames):
        concat = np.concatenate((l, f), axis = 1)
        plotCadence(concat, title="heat flux, temp difference", minVal=-200, maxVal=400) 
        
main()



