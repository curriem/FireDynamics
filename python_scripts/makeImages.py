import numpy as np
import glob
import matplotlib.pyplot as plt
import commands

def main():
    
    # get the file paths to the data you wish to make images out of
    filePaths = glob.glob('/Users/mcurrie/FireStats/data/*npy')
    filePaths = ['/Users/mcurrie/FireStats/data/15fclump1.npy']

    for filePath in filePaths:
        # get rid of the residual images in the images directory
        commands.getoutput('rm /Users/mcurrie/FireStats/data/images/*')
        data = np.load(filePath)
        n = 0
        
        # plot and save the image
        for frame in data:
            plt.figure()
            plt.imshow(frame, interpolation='nearest', cmap='jet', vmin=200, vmax=550)
            plt.colorbar()
            plt.savefig('/Users/mcurrie/FireStats/data/images/im%s.png'%str(n).zfill(4), clobber=True)
            plt.close()
            n+=1
        print 'DONE'
        
        # something to break up the image generation
        answer = raw_input('type y for next file: ')
        if answer == 'y':
            continue
        else:
            break
main()