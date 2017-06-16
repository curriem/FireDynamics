#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:56:09 2017

@author: mcurrie


Want in this code: 
    A histogram that says how many previously on fire neighbors lit a pixel on fire
    A histogram that says how many previously on fire pixels failed to light a pixel on fire
    Burn time distribution

"""

import numpy as np
import matplotlib.pyplot as plt
import glob 

def loadData(filePath):
    """loads data in from a .npy array
    
    Inputs:
    ----------
    
    filePath
        (str) path to .npy file containing data
        
    Returns:
    ----------
    data
        (3d numpy array) containing data with dimensions (time, x, y)
    """
    
    data = np.load(filePath)
    return data

def plotStat(stat, title=None):
    """ plots a timeseries of a given statistic
    
    Inputs:
    ----------
    stat
        (1d numpy array) a timeseries of a given statistic
        
    """
    plt.figure()
    plt.plot(range(len(stat)), stat)
    plt.title(title)
    plt.ylabel('statistic')
    plt.xlabel('time')
    plt.show()

def plotFrame(frame):
    plt.figure()
    plt.imshow(frame, interpolation='nearest', vmin=-1, vmax=1, cmap='jet')
    plt.colorbar()
    #plt.close()

def burnTimes(data):
    
    
    t, x, y = data.shape
    burnTimes = []
    for i in range(x):
        for j in range(y):
            burnTimes.append(np.max(data[:, i, j]))
    
    return burnTimes

def tempStats(data):
    """this gives basic statistics for the data set
    
    Inputs:
    ----------
    
    data
        (3d numpy array) a timeseries of frames with temperature data
    
    Returns:
    ----------
    maxTemp
        (float) the maximum temperature in the dataset
    minTemp
        (float) the minimum temperature in the dataset
    meanTemp
        (float) the mean temperature
    medianTemp
        (float) the median temoerature
    stdTemp
        (float) the standard deviation of the dataset
    
    
    """

    
    # Per Frame:
    maxTemp = np.nanmax(np.nanmax(data, axis=1), axis=1)
    minTemp = np.nanmin(np.nanmin(data, axis=1), axis=1)
    meanTemp = np.nanmean(np.nanmean(data, axis=1), axis=1)
    medianTemp = np.nanmedian(np.nanmedian(data, axis=1), axis=1)
    stdTemp = np.nanstd(np.nanstd(data, axis=1), axis=1)
    
    return maxTemp, minTemp, meanTemp, medianTemp, stdTemp
        
def getNeighbors(x, y, prevFrame, numLayers):
    neighbors = prevFrame[x-numLayers:x+numLayers+1, y-numLayers:y+numLayers+1]

    return neighbors

def makeHist(vector,bins, title, save=False, savePath=None):
    plt.figure()
    plt.hist(vector, bins)
    plt.title(title)
    plt.ylabel('Number of times neighbors didn\'t set a pixel on fire')
    plt.xlabel('Number of neighbors')
    if save:
        plt.savefig(savePath+'.pdf', clobber=True)
    plt.show()

def modelData(data, tempThresh):
    """This models the data with positive integers being on fire (the number 
    corresponding to the burn time), zeros being unburned material, and negative
    ones being burned material
    
    Inputs:
    ----------
    
    data
        (3d numpy array) a timeseries of the data
    
    tempThresh
        (float) the temperature threshild for a pixel being "on fire"
        
    Returns:
    ----------
    
    model
        (3d numpy array) a timeseries for the model of the data
    
    """
    
    data = np.nan_to_num(data)
    model = []
    
    t, x, y = data.shape
    for frame in data:
        modelFrame = np.zeros_like(frame)
        modelFrame[np.where(frame > tempThresh)] = 1
        #plotFrame(modelFrame)
        model.append(modelFrame)
    
    model = np.array(model)
    
    model = np.cumsum(model, axis=0)
    for i in range(x):
        for j in range(y):

            uniqueNums, uniqueInds = np.unique(model[:, i, j], return_index=True)
            model[uniqueInds[-1]+1:, i, j] = -1
    
      
    return model

def countingNeighbors(data, numLayers):
    ''' this counts neighbors in the previous timestep that are on fire in the
    previous frame around a pixel of interest 
    
    Inputs:
    ----------
    
    data
        (3d numpy array) a timeseries of the data
    
    numLayers
        (int) how many neighbors around the pixel of interest you want to look at
        
    Returns:
    ---------
    
    fireCounts
        (1d array) an array of counts of neighbors on fire
    
    '''
    
    t, x, y = data.shape
    fireCounts = []
    for n in range(t)[:-1]:
        
        """
        NOTE: BOUNDARY EFFECT HANDLER IS WRONG!!!! Correcting later. 
        """
        
        onFireX, onFireY = np.where(data[n+1] == 1)
        
        # get rid of boundary effects
        onFireX = onFireX[np.where(onFireX >= numLayers)]
        onFireX = onFireX[np.where(onFireX < x - numLayers)]
        onFireY = onFireY[np.where(onFireY >= numLayers)]
        onFireY = onFireY[np.where(onFireY < y - numLayers)]
        
        unburntX, unburntY = np.where(data[n+1] == 0)
        
        # get rid of boundary effects
        unburntX = unburntX[np.where(unburntX >= numLayers)]
        unburntX = unburntX[np.where(unburntX < x- numLayers)]
        unburntY = unburntY[np.where(unburntY >= numLayers)]
        unburntY = unburntY[np.where(unburntY < y - numLayers)]
        
        
        #for x1, y1 in zip(onFireX, onFireY):
        for x1, y1 in zip(unburntX, unburntY):
            if data[n, x1, y1] == 0:
                neighbors = getNeighbors(x1, y1, data[n], numLayers)
                disregard, fireCount = np.array(np.where(neighbors > 0)).shape
                fireCounts.append(fireCount)
    return np.array(fireCounts)

#==============================================================================
#         for x0, y0 in zip(unburntX, unburntY):
#             print 'hi'
#==============================================================================
            
    
            

def main():
    
    filePaths = glob.glob('/Users/mcurrie/FireStats/data/*npy')
    
    plotTempStats = False
    plotFrames = False
    tempThresh = 400.
    numLayers = 1
    bins = np.arange(1, (2*numLayers+1)**2 + 1, 1)
    
    for filePath in filePaths:
    
        data = loadData(filePath)
        
        model = modelData(data, tempThresh)
        
        savePath = '/Users/mcurrie/FireStats/plots/%s_unlit'%filePath.split('/')[-1].strip('.npy')
        
        if plotTempStats:
        
            maxTemp, minTemp, meanTemp, medianTemp, stdTemp = tempStats(data)
            
            # plotting the statistics
            plotStat(maxTemp, 'max temp, overall max = %f'%maxTemp)
            plotStat(minTemp, 'min temp, overall min = %f'%minTemp)
            plotStat(meanTemp, 'mean temp, overall mean = %f'%meanTemp)
            plotStat(medianTemp, 'median temp, overall median = %f'%medianTemp)
            plotStat(stdTemp, 'std temp, overall std = %f'%stdTemp)
        
        # Make histogram that says how many neighbors lit a pixel on fire
        fireCounts = countingNeighbors(model, numLayers)
        
        fireCounts = fireCounts[np.where(fireCounts > 0)]
        
        makeHist(fireCounts, bins, title=filePath, save=True, savePath = savePath)


main()

