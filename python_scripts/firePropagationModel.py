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

def distanceFromCenterMatrix(numLayers, layerSize):
    """computes a matrix of size one layer around the center pixel
        i.e. if numLayers = 1 then the resulting matrix will be 3x3.
        The matrix comtains distances to the center pixel from each
        location
        
        Inputs:
        ----------
        
        numLayers
            (int) the number of layers around the center pixel you are interested
            in
        layerSize
            (int) the size of the layer (e.g. if numLayers = 1, then layerSize = 3)
        
        Returns:
        ----------
        
        mat
            (2d numpy array) a matrix containing the distance from each cell to the
            center
        
    
    """

    mat = np.empty((layerSize, layerSize))
    center = [numLayers, numLayers]
    for i in range(layerSize):
        for j in range(layerSize):
            mat[i,j] = np.linalg.norm(np.array([i,j]) - np.array(center))
    mat[center[0],center[1]] = np.inf
    
    mat = 1/mat
    return mat

def skewProbabilityMatrix(numLayers, layerSize, skewProb):

    """computes a matrix of size one layer around the center pixel
    i.e. if numLayers = 1 then the resulting matrix will be 3x3.
    The matrix comtains distances to the center pixel from each
    location
    
    Inputs:
    ----------
    
    numLayers
        (int) the number of layers around the center pixel you are interested
        in
    layerSize
        (int) the size of the layer (e.g. if numLayers = 1, then layerSize = 3)
    skewProb
        (1d array) of length two. Specifies how much to skew the probabilities 
        to make it directionally dependent. the values in the array must be between
        -0.5 and 0.5. Usage: if probSkew = [0, 0.5] the probabilities will skew
        to the right and the fire will want to go to the right more. Note that 
        the order is backwards: [y,x] This is an artifact of numpy arrays
    
    Returns:
    ----------
    
    mat
        (2d numpy array) a matrix containing the distance from each cell to the
        center
        
    NOTES:
    ----------
    I should probably find a better way to skew the probabilities but it works
    for now.
        
    
    """
    mat = np.empty((layerSize, layerSize))
    center = [numLayers, numLayers]
    for i in range(layerSize):
        for j in range(layerSize):
            x = i -numLayers
            y = j -numLayers
            mat[i,j] = np.linalg.norm([x - skewProb[0], y - skewProb[1]]) 
    mat[center[0],center[1]] = np.inf
    mat = 1/mat
    return mat

def probThreshMatrix(numLayers, masterProb, layerSize, probSkew):
    """computes a matrix of size one layer around the center pixel
        i.e. if numLayers = 1 then the resulting matrix will be 3x3.
        The matrix contains thresholds that the probaility has to meet
        to light a particular pixel on fire
        
    Inputs:
    ----------
    numLayers
        (int) the numer of layers you are interested in around the center pixel
    prob 
        (float) the chosen probility to light one adjecent pixel on fire
    layerSize
        (int) the size of the layer (e.g. if numLayers = 1, then layerSize = 3)   
        
    Returns:
    ----------
    probThreshMatrix
        (2d numpy array) a matrix containing the probabilities of each cell lighting
        on fire (scaled with distance)
    
    """

    
    #distanceMatrix = distanceFromCenterMatrix(numLayers, layerSize)
    distanceMatrix = skewProbabilityMatrix(numLayers, layerSize, probSkew)
    probThreshMatrix = distanceMatrix*masterProb

    
    return probThreshMatrix
    
def randProbMatrix(numLayers, layerSize):
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
    
    randMatrix = np.random.rand(layerSize, layerSize)
    return randMatrix

def burnTimeGrid(N, homogeneous=False, master=3):
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

    grid = np.ones((N,N))
    grid[N/3:2*N/3, :] = 2
    grid[2*N/3:, :] = 3
    if homogeneous:
        grid = master*np.ones((N,N))
    return grid
    
def main():
    
    plotting=True   # plot at the end?
    save=False       # save the plots?
    N = 100
    masterProb = 0.5
    numLayers = 2
    numTimesteps = 100
    savePath = '/Users/mcurrie/FireDynamics/data/CA/'
    probSkew = [0.0,0.0]
    
    
    # initiate the fire grid
    fireGrid = np.zeros((N, N))
    fireGrid[N/2,N/2] = 1

    
    fireTimeseries = [np.copy(fireGrid)]    # initiate a timeseries for later
    
    layerSize = numLayers*2 +1
    
    burnTimes = burnTimeGrid(N)
    
    for t in range(numTimesteps)[:5]:
        
        # check if any indices burnt out and flag them accordingly
        burntInds = np.array(np.where(fireGrid == burnTimes))
        fireGrid[burntInds[0], burntInds[1]] = -1

        # get the indices that are on fire
        fireInds = np.array(np.where(fireGrid[numLayers:N-numLayers-1, numLayers:N-numLayers-1] > 0))
        fireInds += numLayers
        
        for [x,y] in fireInds.T:

            neighbors = fireGrid[x-numLayers:x+numLayers+1, y-numLayers:y+numLayers+1]

            randProb = randProbMatrix(numLayers, layerSize)

            probThresh = probThreshMatrix(numLayers, masterProb, layerSize, probSkew)
            
            probThresh = (np.ones((layerSize,layerSize)) - np.abs(neighbors))*probThresh
           
            diff = probThresh - randProb
            
            newFireInds = np.array(np.where(diff > 0))
            neighbors[newFireInds[0], newFireInds[1]] = 1
        
            fireGrid[x-numLayers:x+numLayers+1, y-numLayers:y+numLayers+1] = neighbors
            fireGrid[x,y] += 1
        
        # have to do this because python is stupid sometimes 
        temp = np.copy(fireGrid)
        fireTimeseries.append(temp)
        
    fireTimeseries = np.array(fireTimeseries)
    
    if save:
        np.save(savePath+'CA_prob=%s_N=%i_layers=%i.npy'%(str(masterProb), N, numLayers), fireTimeseries)
    
    if plotting:
        for frame in fireTimeseries:
            plt.figure()
            plt.imshow(frame, interpolation='nearest', cmap='jet', vmin=-1, vmax=1)
            plt.axis('off')
            plt.colorbar()

    
main()