#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:56:09 2017

@author: mcurrie


Edited Jul 11 2017

"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.stats as scistats 

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
    
    #CHANGED THIS FROM 
    
    tallyArr0 = np.zeros(8)
    tallyArr1 = np.zeros(8)
    for one, two in zip(onFireUnique, onFireTally):
        tallyArr1[one-1] = two
    for one, two in zip(unburntUnique, unburntTally):
        tallyArr0[one-1] = two
    probs = tallyArr1.astype('float')/(tallyArr1 + tallyArr0)
    
    # below is for spotting 
    
#==============================================================================
#     tallyArr0 = np.zeros(9)
#     tallyArr1 = np.zeros(9)
#     for one, two in zip(onFireUnique, onFireTally):
#         tallyArr1[one] = two
#     for one, two in zip(unburntUnique, unburntTally):
#         tallyArr0[one] = two
#     probs = tallyArr1.astype('float')/(tallyArr1 + tallyArr0)
#==============================================================================
    plt.figure()
    
    # CHANGED THIS FROM plt.plot(np.arange(1,9,1), probs)
    plt.plot(np.arange(1,9,1), probs)
    plt.title(title)
    plt.xlabel('number of neighbors on fire')
    plt.ylabel('probability of lighting on fire')
    if save:
        plt.savefig(savePath)
    plt.show()
    plt.close()
            



def main():
    
    filePaths = glob.glob('/Users/mcurrie/FireStats/data/15fclump2*npy')
    #filePaths = ['/Users/mcurrie/FireStats/data/aggPix/15fopen2_aggLevel10.npy']
    
    plotTempStats = False
    plotFrames = False
    tempThresh = 500.
    numLayers = 1
    bins = np.arange(1, (2*numLayers+1)**2 + 1, 1)
    plotHistsAndProbs = False
    plotBurnTimes = False
    makeTrendline = True
    
    for filePath in filePaths:
        
        
        fireName=filePath.split('/')[-1].strip('.npy')
        
        print fireName 
        
        data = loadData(filePath)
        
        model = modelData(data, tempThresh)
        
        if plotFrames:
            for frame in data:
                plotFrame(frame)
        
        if plotTempStats:
        
            maxTemp, minTemp, meanTemp, medianTemp, stdTemp = tempStats(data)
            
            with open('/Users/mcurrie/FireStats/basicStats.txt', 'a') as file:
                overallMax = np.max(maxTemp)
                overallMin = np.min(minTemp)
                overallMean = np.mean(meanTemp)
                overallMedian = np.mean(medianTemp)
                overallStd = np.mean(stdTemp)
                file.write('%03f\t%03f\t%03f\t%03f\t%03f\n'%(overallMax, overallMin, overallMean, overallMedian, overallStd))
                #file.write('%03d\t%03d\t%03d\t%03d\t%03d\n'%(str(overallMax), str(overallMin), str(overallMean), str(overallMedian), str(overallStd)))

            # plotting the statistics
            #plotStat(maxTemp, 'max temp per frame, %s'%fireName, savePath='/Users/mcurrie/FireStats/plots/%s_maxTemp.pdf'%fireName)
            #plotStat(minTemp, 'min temp per frame, %s'%fireName, savePath='/Users/mcurrie/FireStats/plots/%s_minTemp.pdf'%fireName)
            #plotStat(meanTemp, 'mean temp per frame, %s'%fireName,savePath='/Users/mcurrie/FireStats/plots/%s_meanTemp.pdf'%fireName)
            #plotStat(medianTemp, 'median temp per frame, %s'%fireName,savePath='/Users/mcurrie/FireStats/plots/%s_medianTemp.pdf'%fireName)
            #plotStat(stdTemp, 'std temp per frame, %s'%fireName,savePath='/Users/mcurrie/FireStats/plots/%s_stdTemp.pdf'%fireName)
#==============================================================================
#             plt.figure()
#             time = range(len(maxTemp))
#             plt.plot(time, maxTemp, label='max')
#             plt.plot(time, minTemp, label='min')
#             plt.plot(time, meanTemp, label='mean')
#             plt.plot(time, medianTemp, label='median')
#             plt.plot(time, stdTemp, label='std dev')
#             plt.legend(loc=4)
#             plt.title('Basic Statistics for Fire 15fopen2')
#             plt.xlabel('Frame (time)')
#             plt.ylabel('Temperature (Celcius)')
#             plt.savefig('/Users/mcurrie/FireStats/basicStats.pdf')
#==============================================================================
        
        if plotBurnTimes:
            burnTimes = getBurnTimes(model)
            plt.figure()
            plt.imshow(burnTimes, interpolation='nearest', cmap='jet')
            plt.title('Burn Times Per Pixel for Fire 15fopen2')
            #plt.title('Burn times per pixel, %s, thresh=%i'%(fireName, int(tempThresh)))
            plt.colorbar()
            plt.axis('off')
            plt.savefig('/Users/mcurrie/FireStats/burnTimes.pdf')
            #plt.savefig('/Users/mcurrie/FireStats/plots/%s_%i_burnTimes.pdf'%(fireName, int(tempThresh)))
            plt.close()
        
        if plotHistsAndProbs:
            onFireCounts, unburntCounts = countingNeighbors(model, numLayers)   


            # commented this out for spotting:
            onFireCounts = onFireCounts[np.where(onFireCounts > 0)]        
            unburntCounts = unburntCounts[np.where(unburntCounts > 0)]
            savePath = '/Users/mcurrie/FireStats/histEx10.pdf'
            #savePath = '/Users/mcurrie/FireStats/plots/%s_%i_hists'%(fireName, int(tempThresh))
            #makeHist([onFireCounts, unburntCounts], bins, title='%s, thresh=%i'%(fireName, int(tempThresh)), save=True, savePath = savePath)  
            makeHist([onFireCounts, unburntCounts], bins, title='Neighbor Counts, Aggregation: 10x10', save=True, savePath = savePath)  
            
            savePath = '/Users/mcurrie/FireStats/probsEx10.pdf'
            #savePath = '/Users/mcurrie/FireStats/plots/%s_%i_probs.pdf'%(fireName, int(tempThresh))
            #makeProbPlots(onFireCounts, unburntCounts, title='%s, thresh=%i'%(fireName, int(tempThresh)),save=True, savePath = savePath)
            makeProbPlots(onFireCounts, unburntCounts, title='Probabilities of Combustion, Aggregation: 10x10',save=True, savePath = savePath)
        
        if makeTrendline:
            with open('/Users/mcurrie/FireStats/trendlines.txt', 'a') as file:
                onFireCounts, unburntCounts = countingNeighbors(model, numLayers)   
                onFireCounts = onFireCounts[np.where(onFireCounts > 0)]
                unburntCounts = unburntCounts[np.where(unburntCounts > 0)]
                
                onFireUnique, onFireTally = np.unique(onFireCounts, return_counts=True)
                unburntUnique, unburntTally = np.unique(unburntCounts, return_counts=True)
        
        
                tallyArr0 = np.zeros(8)
                tallyArr1 = np.zeros(8)
                for one, two in zip(onFireUnique, onFireTally):
                    tallyArr1[one-1] = two
                for one, two in zip(unburntUnique, unburntTally):
                    tallyArr0[one-1] = two
                    
                print tallyArr0, tallyArr1
                
                tallyArr0 = np.array([9019.,  5997.,  3533.,  1847.,  1021.,    42.])
                tallyArr1 = np.array([98.,  78.,  63.,  31.,  13.,   7.])
                probs = tallyArr1.astype('float')/(tallyArr1 + tallyArr0)
                
                x = np.arange(1, len(tallyArr0)+1, 1)
                slope, intercept, r_value, p_value, std_err = scistats.linregress(x, probs)
                print slope, intercept, r_value, p_value, std_err
                #file.write('%f\t%f\t%f\t%f\t%f\n'%(slope, intercept, r_value, p_value, std_err))
                

main()

