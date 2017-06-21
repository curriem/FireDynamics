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

def plotStat(stat, title, savePath):
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
    plt.xlabel('frame')
    plt.savefig(savePath)
    plt.show()
    plt.close()

def plotFrame(frame):
    """plots a frame of the data
    
    Inputs:
    ----------
    
    frame
        (2d numpy array) one timestep of the data
    
    """
    plt.figure()
    plt.imshow(frame, interpolation='nearest', vmin=-1, vmax=1, cmap='jet')
    plt.colorbar()
    #plt.close()

def getBurnTimes(data):
    """gets the burn times for each pixel in the time series and plots it
    
    Inputs:
    ----------
    
    data
        (3d numpy array) a timeseries of the data
        
    Returns:
    ----------
    
    burnTimes
        (2d numpy array) a grid of burn times corresponding to each pixel
    
    
    """
    
    t, x, y = data.shape
    burnTimes = np.empty_like(data[0])
    for i in range(x):
        for j in range(y):
            burnTimes[i,j] = np.max(data[:,i,j])
            
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
    """gets the neighbors surrounding a pixel of interest with a specified layer
    
    Inputs:
    ----------
    x
        (int) x coordinate for pixel of interest
    y
        (int) y coordinate for pixel of interest
    prevFrame
        (2d numpy array) the frame immediately before the pixel of interest
    numLayers
        (int) the number of layers you want to calculate the neighbors for 
        
    Returns:
    ----------
    
    neighbors
        (2d numpy array) the neighbors surrounding the pixel of interest
    
    """
    neighbors = prevFrame[x-numLayers:x+numLayers+1, y-numLayers:y+numLayers+1]

    return neighbors

def makeHist(vector ,bins, title, save=False, savePath=None):
    """ makes a histogram for the number of neighbors surrounding a pixel that 
    were on fire in the previous frame
    
    Inputs:
    ----------
    
    vector
        (1d numpy array) array containing counts on neighbors on fire
    bins
        (1d numpy array) array of specified bins for the histogram
    title
        (str) the title for the histogram
    save (optional)
        (bool) want to save the histogram?
    savePath 
        (str) path to save plot (must be specified if save=True)
    
    
    """
    
    onFireCounts = vector[0]
    unburntCounts = vector[1]
    plt.figure()
    plt.hist(np.concatenate((unburntCounts, onFireCounts)), bins, label="neighbors did NOT light pixel")
    plt.hist(onFireCounts, bins, label="neighbors DID light pixel")
    plt.title(title)
    plt.legend()
    plt.ylabel('Number of instances')
    plt.xlabel('Number of neighbors')
    if save:
        plt.savefig(savePath+'.pdf', clobber=True)
    plt.show()
    plt.close()

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
    onFireCounts = []
    unburntCounts = []
    for n in range(t)[:-1]:
                
        onFireX, onFireY = np.where(data[n+1] == 1)
        unburntX, unburntY = np.where(data[n+1] == 0)
           
        for x1, y1 in zip(onFireX, onFireY):
            if data[n, x1, y1] == 0:
                neighbors = getNeighbors(x1, y1, data[n], numLayers)
                if neighbors.shape != (2*numLayers+1, 2*numLayers+1):
                    continue
                disregard, fireCount = np.array(np.where(neighbors > 0)).shape
                onFireCounts.append(fireCount)
            
        for x1, y1 in zip(unburntX, unburntY):
            if data[n, x1, y1] == 0:
                neighbors = getNeighbors(x1, y1, data[n], numLayers)
                if neighbors.shape != (2*numLayers+1, 2*numLayers+1):
                    continue
                disregard, fireCount = np.array(np.where(neighbors > 0)).shape
                unburntCounts.append(fireCount)
                
    return np.array(onFireCounts), np.array(unburntCounts)


  
def makeProbPlots(onFireCounts, unburntCounts, title=None, save=False, savePath=None):
    """makes plots for the probability of a cell lighting of fire with a certain
    number of neighbors
    
    Inputs:
    ----------
    
    onFireCounts
        (1d numpy array) an array of neighbor counts for cells that lit on fire        
    unburntCounts 
        (1d numpy array) an array of neighbor counts for cells that did not light on fire
    title
        (str) a title for the plot
    save
        (bool) save the plot?
    savePath
        (str) if save is true, specify the path where you want to save the plot
    
    
    """
    
    onFireUnique, onFireTally = np.unique(onFireCounts, return_counts=True)
    unburntUnique, unburntTally = np.unique(unburntCounts, return_counts=True)
    tallyArr0 = np.zeros(8)
    tallyArr1 = np.zeros(8)
    for one, two in zip(onFireUnique, onFireTally):
        tallyArr1[one -1] = two
    for one, two in zip(unburntUnique, unburntTally):
        tallyArr0[one -1] = two
    probs = tallyArr1.astype('float')/(tallyArr1 + tallyArr0)
    plt.figure()
    plt.plot(np.arange(1,9,1), probs)
    plt.title(title)
    plt.xlabel('number of neighbors on fire')
    plt.ylabel('probability of lighting on fire')
    if save:
        plt.savefig(savePath)
    plt.show()
    plt.close()
            

def main():
    
    filePaths = glob.glob('/Users/mcurrie/FireStats/data/*npy')
    
    plotTempStats = False
    plotFrames = False
    tempThresh = 500.
    numLayers = 1
    bins = np.arange(1, (2*numLayers+1)**2 + 1, 1)
    plotHistsAndProbs = True
    plotBurnTimes = True
    
    for filePath in filePaths:
        
        fireName=filePath.split('/')[-1].strip('.npy')
        
        data = loadData(filePath)
        
        model = modelData(data, tempThresh)
        
        if plotFrames:
            for frame in data:
                plotFrame(frame)
        
        if plotTempStats:
        
            maxTemp, minTemp, meanTemp, medianTemp, stdTemp = tempStats(data)
            
            # plotting the statistics
            plotStat(maxTemp, 'max temp per frame, %s'%fireName, savePath='/Users/mcurrie/FireStats/plots/%s_maxTemp.pdf'%fireName)
            plotStat(minTemp, 'min temp per frame, %s'%fireName, savePath='/Users/mcurrie/FireStats/plots/%s_minTemp.pdf'%fireName)
            plotStat(meanTemp, 'mean temp per frame, %s'%fireName,savePath='/Users/mcurrie/FireStats/plots/%s_meanTemp.pdf'%fireName)
            plotStat(medianTemp, 'median temp per frame, %s'%fireName,savePath='/Users/mcurrie/FireStats/plots/%s_medianTemp.pdf'%fireName)
            plotStat(stdTemp, 'std temp per frame, %s'%fireName,savePath='/Users/mcurrie/FireStats/plots/%s_stdTemp.pdf'%fireName)
        
        if plotBurnTimes:
            burnTimes = getBurnTimes(model)
            plt.figure()
            plt.imshow(burnTimes, interpolation='nearest', cmap='jet')
            plt.title('burn times per pixel, %s, thresh=%i'%(fireName, int(tempThresh)))
            plt.colorbar()
            plt.axis('off')
            plt.savefig('/Users/mcurrie/FireStats/plots/%s_%i_burnTimes.pdf'%(fireName, int(tempThresh)))
            plt.close()
        
        if plotHistsAndProbs:
            onFireCounts, unburntCounts = countingNeighbors(model, numLayers)        
            onFireCounts = onFireCounts[np.where(onFireCounts > 0)]        
            unburntCounts = unburntCounts[np.where(unburntCounts > 0)]
            savePath = '/Users/mcurrie/FireStats/plots/%s_%i_hists'%(fireName, int(tempThresh))
            makeHist([onFireCounts, unburntCounts], bins, title='%s, thresh=%i'%(fireName, int(tempThresh)), save=True, savePath = savePath)        
            savePath = '/Users/mcurrie/FireStats/plots/%s_%i_probs'%(fireName, int(tempThresh))
            makeProbPlots(onFireCounts, unburntCounts, title='%s, thresh=%i'%(fireName, int(tempThresh)),save=True, savePath = savePath)
        
        
main()

