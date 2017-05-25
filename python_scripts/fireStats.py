#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:56:09 2017

@author: mcurrie
"""

import numpy as np
import matplotlib.pyplot as plt

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

def plotStat(stat, title=None, save=False, savePath=None):
    """ plots a timeseries of a given statistic
    
    Inputs:
    ----------
    stat
        (1d numpy array) a timeseries of a given statistic
    title
        (str) title for the plot
    save
        (bool) if true, it saves the plot
    savePath
        (str) path to save to 
        
    """
    plt.figure()
    plt.plot(range(len(stat)), stat)
    plt.title(title)
    plt.ylabel('Statistic')
    plt.xlabel('Frame')
    if save:    
        plt.savefig(savePath)
    plt.show()
    

def plotFrame(frame, vmin=-1, vmax=1):
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
    plt.colorbar()
    plt.close()

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
    
    # Entire dataset:
    maxTemp = np.nanmax(data)
    minTemp = np.nanmin(data)
    meanTemp = np.nanmean(data)
    medianTemp = np.nanmedian(data)
    stdTemp = np.nanstd(data)
    
    # Per Frame:
    maxTempTimeseries = np.nanmax(np.nanmax(data, axis=1), axis=1)
    minTempTimeseries = np.nanmin(np.nanmin(data, axis=1), axis=1)
    meanTempTimeseries = np.nanmean(np.nanmean(data, axis=1), axis=1)
    medianTempTimeseries = np.nanmedian(np.nanmedian(data, axis=1), axis=1)
    stdTempTimeseries = np.nanstd(np.nanstd(data, axis=1), axis=1)
    
    return maxTemp, maxTempTimeseries, minTemp, minTempTimeseries, \
        meanTemp, meanTempTimeseries, medianTemp, medianTempTimeseries, \
        stdTemp, stdTempTimeseries
        
def getNeighbors(prevFrame, newFireCoords, numLayers):
    """ this gets the number of surrounding neighbors that were on fire in the 
    previous frame within a certain number of layers around the pixel of interest
    
    Inputs:
    ----------
    
    prevFrame
        (2d numpy array) the frame before the current frame of interest
    
    newFireCoords
        (2d numpy array) an array containing the coordinates [x,y] of pixels 
        that are newly on fire in the current frame
    
    numLayers
        (int) the number of layers of neighbors you are interested in
        
    Returns:
    ----------
    
    counts
        (1d array) for each pixel newly on fire, this contains the number of 
        neighbor pixels that were on fire in the previous frame
    
    """
    
    counts = []
    for [x,y] in newFireCoords.T:
        neighbors = prevFrame[x-numLayers:x+numLayers+1, y-numLayers:y+numLayers+1]
        count =  np.array(np.where(neighbors > 0)).shape[1]
        counts.append(count)
    return counts

def makeHist(counts, numBins, save=False, savePath=None):
    """makes a histogram of the number of occurances of the number of neighbors
    that were on fire in the previous frame
    
    Inputs:
    ----------
    
    counts
        (1d array) an array containing the number of neighbors that were on fire
        in the previous frame for each newly on fire pixel
        
    numBins
        (int) the number of bins to produce
        
    save
        (bool) true if you want to save the plot. Default is false
        
    savePath
        (str) the path to the file name you want to save under (excluding the 
        file type e.g. ".pdf")
    
    """
    
    plt.figure()
    plt.hist(counts, numBins)
    if save:
        plt.savefig(savePath+'.pdf', clobber=True)
    plt.show()

def main():
    fireNum = 3
    filePath = '/Users/mcurrie/FireDynamics/data/f%i/f%i_dataCube.npy'%(fireNum, fireNum)
    #filePath = '/Users/mcurrie/FireDynamics/data/CA/CA_prob=0.5_N=1000_lag=3_layers=2.npy'
    #filePath = '/Users/mcurrie/FireDynamics/data/CA/CA_prob=0.5_N=1000_lag=2_layers=2.npy'
    savePath = '/Users/mcurrie/repositories/FireDynamics/plots/'
    data = loadData(filePath)
    realData=True
    plotting=True
    save=True
    tempThresh = 500.
    
    
    if realData:
    
        maxTemp, maxTempTimeseries, minTemp, minTempTimeseries, \
            meanTemp, meanTempTimeseries, medianTemp, medianTempTimeseries, \
            stdTemp, stdTempTimeseries = tempStats(data)
        print 'Max Temp:', maxTemp
        print 'Min Temp:', minTemp
        print 'Mean Temp:', meanTemp
        print 'Median Temp:', medianTemp
        print 'Standard Dev:', stdTemp
        # plotting the statistics
        if plotting:
            plotStat(maxTempTimeseries, 'Maximum Temperature by Frame', save, savePath+'f%i_maxtemp.pdf'%fireNum)
            plotStat(minTempTimeseries, 'Minimum Temperature by Frame'%minTemp, save, savePath+'f%i_mintemp.pdf'%fireNum)
            plotStat(meanTempTimeseries, 'Mean Temperature by Frame'%meanTemp,save, savePath+'f%i_meantemp.pdf'%fireNum)
            plotStat(medianTempTimeseries, 'Median Temperature by Frame'%medianTemp, save, savePath+'f%i_mediantemp.pdf'%fireNum)
            plotStat(stdTempTimeseries, 'Standard Deviation of Temperature by Frame'%stdTemp, save, savePath+'f%i_stdtemp.pdf'%fireNum)
    
    # Make histogram
    else:
        numLayers = 2
        numBins = (2*numLayers+1)**2
        counts = np.array([0])
        for n in range(len(data))[:-1]:
            if plotting:
                plotFrame(data[n])
                newFireCoords = np.array(np.where(data[n+1] == 1))
                neighborsOnFire = getNeighbors(data[n], newFireCoords, numLayers)
                counts = np.concatenate((counts, neighborsOnFire))
                
                assert len(counts) < data[0].shape[0]**2
                
        makeHist(counts, numBins)#, save=True, savePath = '/Users/mcurrie/FireDynamics/data/CA/test')

main()

