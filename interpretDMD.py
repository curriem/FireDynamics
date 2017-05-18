"""
EXAMPLE USAGE FOR THIS MODULE:

def main():
    filePaths = glob.glob("/Users/mcurrie/FireDynamics/data/DMD_reconstruction/ReducedSolutions/Reduced_solution_3*.csv")
    maskPath = "/Users/mcurrie/FireDynamics/data/13fsingle3/mask3.npy"
    mask = np.load(maskPath)
    mask = mask.astype('int') # changes from bool type to int
    
    for filePath in filePaths:
        print filePath
        saveTag = createSaveTag(filePath)
        saveDataCubePath = "/Users/mcurrie/FireDynamics/data/DMD_reconstruction/ReducedSolutions/dataCube%s.npy"%saveTag
      

        data = loadData(filePath)
        dataCube = reconstructImages(data, mask)
        np.save(saveDataCubePath, dataCube)
        
        makePNGs(dataCube, saveTag)
                
"""




import numpy as np
import matplotlib.pyplot as plt
import glob
import os



def loadData(filePath):
    """ Loads data into numpy array from a csv file
    
    Inputs:
    ----------
    
    filePath
        (str) a string containing the path to a .csv file 
        
    Returns:
    ----------
    
    data
        (2d numpy array) a numpy array containing the data with dimensions 
        (timesteps, temperature data)
    
    """
    
    data = np.loadtxt(filePath)
    return data.T

def reconstructImages(data, mask):
    """ Reconstructs a timeseries of images from the 2d reduced solution array
    
    Inputs:
    ----------
    
    data
        (2d numpy array) a numpy array containing the data with dimensions 
        (timesteps, temperature data)
    mask
        (2d numpy array) a mask of what an image from a single timestep should
        look like. Ones where data goes, padded with zeros to make it a rectangular 
        matrix
        
    Returns:
    ----------
    
    dataCube
        (3d numpy array) a timeseries of images containing the temperature data
        with dimenstions (time, length, width)
    
    
    """
    nonZeroInds = np.array(np.nonzero(mask))
    nonZeroInds = nonZeroInds.T
    dataCube = []
    
    for cadence in data:
        aperture = np.empty_like(mask)
        aperture[nonZeroInds[:, 0], nonZeroInds[:, 1]] = cadence
        dataCube.append(aperture)
    
    dataCube = np.array(dataCube)
    return dataCube
    
  
def makeImages(dataCube, saveTag, savePath, imType='png'):
    """Makes pdfs or pngs of the image at each timestep and puts them in a 
    corresponding directory Note: makes pngs by default
    
    Inputs:
    ----------
    
    dataCube
        (3d numpy array) a timeseries of images containing the temperature data
        with dimenstions (time, length, width)
    saveTag
        (str) naming convention for files
        example: if saveTag = "3_10" then the images will be saved in a directory "images_3_10/"
    savePath
        (str) path to directory where you want to put a directory containing the 
        resulting images
    imType
        (str) specify what type of image you want to create. Handles either png 
        or pdf (png by default)
    
    """
    
    minVal = 0
    maxVal = 900
    n = 1
    imDir = '/images_%s/'%saveTag
    for cadence in dataCube:
        imName = 'im%s.%s'%(str(n).zfill(4), imType)
        plt.figure()
        plt.imshow(cadence, cmap='jet', vmin=minVal, vmax=maxVal)
        plt.axis('off')
        plt.colorbar()        
        try:
            plt.savefig(savePath+imDir+imName, bbox_inches='tight', clobber=True)
        except IOError:
            os.mkdir(savePath+imDir)
            plt.savefig(savePath+imDir+imName, bbox_inches='tight', clobber=True)
        plt.close()
        n += 1
        
def createSaveTag(filePath):
    """ Creates a naming convention for the reduced solutions
    
    Inputs:
    ----------
    filePath
        (str) the path to the csv file of interest
        
    Returns:
    ----------
    
    saveTag
        (str) a tag to put on the end of files to differentiate datasets
    
    """
    
    saveTag = filePath.split('/')[-1].strip("Reduced_solution_").strip(".csv")
    return saveTag


