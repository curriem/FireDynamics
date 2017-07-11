#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:13:40 2017

@author: mcurrie
"""

import numpy as np
import math
import glob

def aggregatePixels(frame, aggLevel):
    ''' Takes groups of pixels and aggregates them into one
    
    Inputs:
    ----------
    
    frame
        (2d numpy array) frame the user wishes to perform aggrecation on
        
    aggLevel
        (int) the size of the groups that the script will use for aggregation
        ex: aggLevel = 3 means the script will take a 3x3 group and make it into one pixel
        
    Returns:
    ----------
    
    newFrame
        (2d numpy array) the new, aggregated frame
        
    
    '''
    
    newFrame = []
    x,y = frame.shape
    for i in range(x)[0::aggLevel]:
        for j in range(y)[0::aggLevel]:
            newFrame.append(np.max(frame[i:i+aggLevel, j:j+aggLevel]))
    newFrame = np.array(newFrame)
    newFrame = newFrame.reshape((int(math.ceil(float(x)/aggLevel)), int(math.ceil(float(y)/aggLevel))))
    return newFrame

def main():
    
    # paths to the files the user wants to aggregate
    filePaths = glob.glob('/Users/mcurrie/FireStats/data/*npy')
    
    aggLevels = [2,3,5,10] # aggregations levels to try
    for filePath in filePaths[0::4]:
        data = np.load(filePath)

        for aggLevel in aggLevels:
            savePath = filePath.strip('.npy')+'_aggLevel%i.npy'%aggLevel
            print savePath
            newData = []
            for frame in data:
                newFrame = aggregatePixels(frame, aggLevel)
                newData.append(newFrame)
            
            newData = np.array(newData)
            np.save(savePath, newData)
            

    
main()
