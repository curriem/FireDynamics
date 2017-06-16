#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:00:35 2017

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
                
def realInitGrid(filePath):
    tempThresh = 500.
    data = np.load(filePath)
    initGrid = np.zeros_like(data[0])
    initGrid[np.where(data[0] > tempThresh)] = 1
    
    return initGrid
    
def padwithnum(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 1
    vector[-pad_width[1]:] = 1
    return vector

def padData(data):
    padParam = 10
    t, x, y = data.shape
    data = np.nan_to_num(data)
    padded = np.zeros((t, x+padParam, y+padParam))
    padded[:, padParam/2:x+padParam/2, padParam/2:y+padParam/2] = data
    return padded

def getProbs(frame, nextFrame, numLayers):
    
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

    N = 50
    numLayers = 1
    masterBurnTime = 30
    numTimesteps = 10
    
    realDataPath = '/Users/mcurrie/FireDynamics/data/f3/f3_dataCube.npy'
    savePath = '/Users/mcurrie/FireDynamics/data/CA/'
    filePath = '/Users/mcurrie/FireDynamics/data/f3/f3_dataCube.npy'
    
    
    #fireGrid = realInitGrid(filePath)


    fireGrid = initialFire(N, 'point')
    data = np.load(realDataPath)
    data = np.nan_to_num(data)
    tempThresh = 500.

   # model = modelData(data, tempThresh)
    
  #  burnTimeGrid = masterBurnTime*np.ones_like(model[-1])
    
   # burnTimeGrid[np.where(model[-1] == 0)] = 0
    
    burnTimes = burnTimeGrid(N, masterBurnTime, homogeneous=True)
    
    fireTimeseries = [np.copy(fireGrid)]    # initiate a timeseries for later
    
    #c
    
    for t in range(numTimesteps):
        #print "COUNTER:", t+1
        
        onFireInds = np.where(fireGrid > 0)
        
        fireGrid = realProbs(fireGrid)
        fireGrid[onFireInds] += 1

        # check to see if any pixels are going to burn out and flag them accordingly
        burntInds = np.array(np.where(fireGrid > burnTimes))
        fireGrid[burntInds[0], burntInds[1]] = -1

        # have to do this because python is stupid sometimes 
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
