#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:00:35 2017

Edited on Tue Jun 11 12:36:41 2017

@author: mcurrie


key:
-1 = has been on fire and can no longer ignite
0 = not on fire and can ignite at a later time
1 = on fire for one time step
2 = on fire for two time steps
3 = on fire for three time steps
"""

import numpy as np 
import matplotlib.pyplot as plt


def plotFrame(frame, vmin=-1, vmax=1, title=None):
    """ plots a frame 
    
    Inputs:
    ----------
    
    frame
        (2d numpy array) a frame to be plotted
        
    vmin 
        (int) the minimum value to plot with the color bar. Default is -1 
    
    vmax 
        (int) the maximum value to plot with the color bar. Default is 1
    """
    
    plt.figure()
    plt.imshow(frame, interpolation='nearest', vmin=vmin, vmax=vmax, cmap='jet')
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    #plt.close()
    
def probMatrix(numLayers, masterProb, windMag, windDir):
    """ computes a matrix of probabilities for the immediate neighbors of a 
    pixel of interest. The probabilities depend on distance from the center
    pixel, the master probability, and the wind speed and direction
    
    Inputs:
    ----------
    
    numLayers
        (int) the numer of layers you are interested in around the center pixel     
        
    masterProb
        (float) the probability of a pixel immediately to the north, south, east, 
        or west lighting on fire with no wind
        
    windMag
        (float) the strength of the wind. Must be less than masterProb and 1-masterProb
        
    windDir
        (float) the angle of the wind measured clockwise from the north (in radians)
            
     
    Returns:
    ----------
    
    probMat
        (2d numpy array)  a matrix containing the probabilities of the neighbors
        lighting on fire
    """
    
    assert windMag <= np.min([masterProb, 1 - masterProb]), \
                'The magnitude of the wind must be less than masterProb and 1-masterProb!'
                
    matDim = numLayers*2 + 1  # the dimensions of the matrix of neighbors

    # Compute the distance scales         
    mat = np.empty((matDim, matDim))
    center = [numLayers, numLayers]
    for i in range(matDim):
        for j in range(matDim):
            mat[i,j] = np.linalg.norm(np.array([i,j]) - np.array(center))
    mat[center[0],center[1]] = np.inf    
    mat = 1/mat
    
    # Add wind
    windMat = np.empty_like(mat)
    for i in range(matDim):
        for j in range(matDim):
            if i == j == numLayers:
                windMat[i,j] = 0
                continue
            pixVector = np.array([(j - center[1]), -(i - center[0])])
            windVector = np.array([np.sin(windDir), np.cos(windDir)])
            cosAlpha = np.dot(pixVector, windVector)/ \
                    (np.linalg.norm(pixVector) * np.linalg.norm(windVector))
            windMat[i,j] = cosAlpha
    windMat *= windMag        
    
    probMat = mat*masterProb + windMat
    probMat[center[0], center[1]] = 0
    
    negInds = np.array(np.where(probMat < 0))
    probMat[negInds[0], negInds[1]] = 0
    
    return probMat*0.1
    

    

def randProbMatrix(numLayers):
    """computes a matrix of random values of size one layer around the center pixel
        i.e. if numLayers = 1 then the resulting matrix will be 3x3.
        
        Inputs:
        ----------   
        
        numLayers
            (int) the numer of layers you are interested in around the center pixel
        
        Returns:
        ----------
        
        randMatrix
            (2d numpy array) a matrix with random values from 0 to 1.0
    """
    
    matDim = 2*numLayers + 1
    randMatrix = np.random.rand(matDim, matDim)
    return randMatrix

def burnTimeGrid(N, master, homogeneous=False):
    """ this produces a custom grid of regions with varying burn times in the cells.
    By default it trisects the entire grid with burn times of one in the bottom 
    third, two in the middle third, and three in the top third
    
    Inputs:
    ----------
    
    N
        (int) the length of a side of the grid
    homogeneous
        (bool) if true, this function produces a homogeneous grid specified by
        the master value
    master
        (int) the master burn time that is used if homogeneous=True
        
    Returns:
    ----------
    
    burnTimeGrid
        (2d numpy array) a grid of varying burn times
        
    """

    grid = np.ones((N, N))
    #grid[N/3:2*N/3, :] = 2
    #grid[2*N/3:, :] = 3
    if homogeneous:
        grid = master*grid
    
    return grid
    
def initialFire(N, fireType):
    '''This sets up an initial fire from which the propagation will start. 
    
    Inputs:
    ----------
    
    N
        (int) the length of a side of the grid
    
    fireType
        (str) the type of fire the user wants to start with. Can be 'line', 
            'parabola', or 'point'. Any deviation defaults to 'point'.
    
    
    '''
    
    
    fireGrid = np.zeros((N, N))
    
    if fireType == 'line':
        fireGrid[N/2,N/3:2*N/3] = 1
    elif fireType == 'point':
        fireGrid[N/2, N/2] = 1
    elif fireType == 'parabola':
        x = np.arange(N/3, 2*N/3)
        y = 0.01*(x-N/2+1)**2 + N/2
        fireGrid[np.rint(y).astype('int'),np.rint(x).astype('int')] = 1
    else:
        print 'Specified fire type not recognised. Proceeding with a point fire.'
        fireGrid[N/2, N/2] = 1
        
    return fireGrid

def realProbs(frame):
    ''' This function hard-codes some probabilities derived from the datasets 
    13fsingle*.npy. 
    
    
    Inputs:
    ----------
    
    frame
        (2d numpy array) the starting frame for propagation calculations for this 
        timestep
        
    Returns:
    ----------
    
    newFrame 
        (2d numpy array) the new frame for the next timestep with propagations calculated and plotted
    '''
    
    realProbs = [0.0784, 0.104, 0.0915, 0.197, 0.235, 0.251, 0.226, 0.305]
    newFrame = np.copy(frame)
    x, y = frame.shape
    for i in range(x):
        for j in range(y):
            cell = frame[i,j]
            neighbors = frame[i-1:i+2, j-1:j+2]
            disregard, fireCount = np.array(np.where(neighbors>0)).shape
            if cell == 0 and fireCount > 0:

                prob = realProbs[fireCount -1]
                randomNum = np.random.rand()
                if randomNum < prob:
                    newFrame[i,j] = 1
    #plotFrame(newFrame)
    return newFrame
                
def realInitGrid(filePath, tempThresh):
    '''This function incorporates real data into the initial fire. It takes the first frame
    of a real dataset, models it, and returns it for the initial propagation frame in CA
    
    Inputs:
    ----------
    
    filePath
        (str) the path to the real data you wish to use as the initial fire
        
    Returns:
    ----------
    
    initGrid
        (2d numpy array) an initial fire grid based on real data
    
    '''
    data = np.load(filePath)
    initGrid = np.zeros_like(data[0])
    initGrid[np.where(data[0] > tempThresh)] = 1
    
    return initGrid
    
def padwithnum(vector, pad_width, iaxis, kwargs):
    '''pads a 2d numpy array with a number of the user's choice
    
    Inputs:
    ----------
    
    vector
        (ndarray) some vector the user wished to pad
        
    pad_with
        (int) the number the user wishes to pad with
        
    Returns:
    ----------
    
    vector
        (ndarray) the padded vector
        
        
    
    '''
    
    vector[:pad_width[0]] = 1
    vector[-pad_width[1]:] = 1
    return vector

def padData(data):
    '''Alternate method of padding a 3d array
    
    Inputs:
    ----------
    data
        (3d numpy array) the data of interest
        
    Returns:
    ----------
    
    padded 
        (3d numpy array) the data, padded
    
    '''
    
    padParam = 10           # the number the user wishes to pad with
    t, x, y = data.shape 
    data = np.nan_to_num(data)
    padded = np.zeros((t, x+padParam, y+padParam))
    padded[:, padParam/2:x+padParam/2, padParam/2:y+padParam/2] = data
    return padded

def getProbs(frame, nextFrame, numLayers):
    '''Generates the probabilities for real datasets
    
    Inputs:
    ----------
    
    frame
        (2d numpy array) frame for this timestep
        
    newFrame
        (2d numpy array) frame for next timestep
        
    numLayers
        (int) the number of layers that the fire is allowed to skip
        
    Returns:
    ----------
    
    (litCount, unlitCount)
        (tuple):
            
            litCount
                the number of neighbors in the previous timestep that were on fire when the pixel of interest lit on fire
            
            unlitCount
                the number of neighbors in the previous timestep that were on fire when the pixel of interest DID NOT light on fire
            
    
    '''
    
    
    x, y, = frame.shape
    litCount = []
    unlitCount = []
    for l in range(numLayers):
        litCount.append([0])
        unlitCount.append([0])
        
    for i in range(x):
        for j in range(y):
            cell = frame[i,j]
            for numLayers in range(numLayers):
                numLayers+=1
                baseNeighborMask = np.zeros((2*numLayers -1, 2*numLayers -1))
                neighbors = frame[i-numLayers:i+numLayers+1, j-numLayers:j+numLayers+1]
                neighborMask = np.lib.pad(baseNeighborMask, 1, padwithnum)
                if neighbors.shape != neighborMask.shape:
                    continue
                neighbors *= neighborMask
                nextCell = nextFrame[i,j]
                if cell == 0:
                    disregard, neighborCount = np.array(np.where(neighbors>0)).shape
                    if nextCell == 1:
                        litCount[numLayers-1].append(neighborCount)
                    elif nextCell == 0:
                        unlitCount[numLayers-1].append(neighborCount)
    return np.array(litCount), np.array(unlitCount)





def main():
    
    plotting=True  # plot at the end?
    save=False       # save the plots?

    N = 50                     # the size of grid you wish to calculate for (this script will make an NxN grid)
    numLayers = 1              # the number of layers that the fire can skip
    masterBurnTime = 30        # the number of timesteps that a pixel is allowed to stay on fire
    numTimesteps = 100         # the number of timesteps the user wishes to calculate for
    
    realDataPath = '/Users/mcurrie/FireDynamics/data/f3/f3_dataCube.npy'
    savePath = '/Users/mcurrie/FireDynamics/data/CA/'                     
    filePath = '/Users/mcurrie/FireDynamics/data/f3/f3_dataCube.npy'
    
    


    fireGrid = initialFire(N, 'point')      # initiate the fire
    data = np.load(realDataPath)
    data = np.nan_to_num(data)              # changes NaNs to 0
    tempThresh = 500.                       # temp that a cell is considered on fire
    #fireGrid = realInitGrid(filePath, tempThresh)

   # model = modelData(data, tempThresh)
    
  #  burnTimeGrid = masterBurnTime*np.ones_like(model[-1])
    
   # burnTimeGrid[np.where(model[-1] == 0)] = 0
    
    burnTimes = burnTimeGrid(N, masterBurnTime, homogeneous=True)  # generates a grid of burn times (doesn't have to be homogeneous)
    
    fireTimeseries = [np.copy(fireGrid)]    # initiate a timeseries for later
    
    
    for t in range(numTimesteps):
        
        onFireInds = np.where(fireGrid > 0)  # gets the indices of the pixels that are on fire
        
        fireGrid = realProbs(fireGrid)       # gets a new frame with new pixels that are on fire
        fireGrid[onFireInds] += 1            # adds one to the pixels taht are on fire

        # check to see if any pixels are going to burn out and flag them accordingly
        burntInds = np.array(np.where(fireGrid > burnTimes))
        fireGrid[burntInds[0], burntInds[1]] = -1

        # have to do this because of memory allocation in python 
        temp = np.copy(fireGrid)
        fireTimeseries.append(temp)
        
    fireTimeseries = np.array(fireTimeseries)
    
    if save:
        #np.save(savePath+'CA_type=%s_layers=%i_burnTime=%i_wind=%s.npy'%(fireType, numLayers, masterBurnTime, str(windMag)), fireTimeseries)
        np.save(savePath+'CA_real_analog_f3.npy', fireTimeseries)       
        
    if plotting:
        n = 1
        for frame in fireTimeseries:
            plt.figure()
            plt.imshow(frame, interpolation='nearest', cmap='jet', vmin=-1, vmax=1)
            plt.axis('off')
            plt.colorbar()
            if save:
                plt.savefig(savePath+'/images/im%s.png'%str(n).zfill(4), clobber=True)
            #plt.close()
            n += 1
    
main()
