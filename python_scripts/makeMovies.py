#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:08:34 2017

Edited Jul 11 2017

@author: mcurrie
"""
import glob
import matplotlib.pyplot as plt
import commands, os
import numpy as np

def main():
    
    
    # paths to the .npy files containing the data the user wished to make the movie out of 
    filePaths = glob.glob('/Users/mcurrie/FireStats/data/ssingle*aggLevel3**npy')
    filePaths = glob.glob('/Users/mcurrie/FireStats/DMD/13fsingle3*reducedSol*npy')
    
    for filePath in filePaths:
        
        # clear the image directory of any residual images
        commands.getoutput('rm /Users/mcurrie/FireStats/DMD/images/*')
        
        # load the data and get rid of NaNs
        data = np.load(filePath)
        data = np.nan_to_num(data)
        
        
        n = 1
        
        # plot the data and save the figure
        for frame in data:
            plt.figure()
            plt.imshow(frame, interpolation='nearest', cmap='jet', vmin = 200, vmax=500)#, vmin=int(math.ceil(np.min(data) / 100.0)) * 100, vmax=int(math.ceil(np.max(data) / 100.0))*100)
            plt.colorbar()
            plt.axis('off')
            plt.savefig('/Users/mcurrie/FireStats/DMD/images/im%s.png'%str(n).zfill(4), clobber=True)
            plt.close()
            n+=1
            
        
        os.chdir('/Users/mcurrie/FireStats/DMD/images/')
        fireName  = filePath.split('/')[-1].strip('.npy')
        
        # create the movie
        commands.getoutput('ffmpeg -r 10 -i im%04d.png /Users/mcurrie/FireStats/DMD/'+fireName+'.mp4')
        os.chdir('/Users/mcurrie/FireStats/DMD/')
        
main()
        