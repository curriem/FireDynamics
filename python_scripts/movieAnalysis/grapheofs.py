#####Graphs the PCA from a videofs.py's anaylsis#####
####Things to change if this file is to be used for a different video or a different computer are marked with #*#. ####
import numpy as np
import matplotlib.pyplot as plt

####Loads the numpy array with the PCA data####
with np.load('PCA.npz') as data:                                       #*#
	pca = data['pca']
	eigen = data['eigen']


####Separates out each separate EOF and eigenvalue to graph####
UU1 = pca.real[0,:,:]
VV1 = pca.imag[0,:,:]
eig1 = np.array_str(eigen[0])
UU2 = pca.real[1,:,:]
VV2 = pca.imag[1,:,:]
eig2 = np.array_str(eigen[1])
UU3 = pca.real[2,:,:]
VV3 = pca.imag[2,:,:]
eig3 = np.array_str(eigen[2])
UU4 = pca.real[3,:,:]
VV4 = pca.imag[3,:,:]
eig4 = np.array_str(eigen[3])
y,x = np.mgrid[0:89,0:159]                                            #*# The grid will have to match the size of the numpy arrays. 


####Graphs and saves each of the sets of data####                     #*# Each of the labels and save destinations may need to be saved due to the frames chosen and file location. Make sure to save the four different graphs to different names or only the last one will be saved. 
plt.figure()
plt.streamplot(x,y,UU1,VV1,cmap='nipy_spectral')
plt.suptitle("PCA of Fire Video Analysis frames 2118-2358. Associated Percent Variance: ")
plt.title(eig1, fontsize=8)
plt.savefig("PCA_of_Fire_Vid_Analysis2118-2358-1.png")
#plt.close()
plt.figure()
plt.streamplot(x,y,UU2,VV2,cmap='nipy_spectral')
plt.suptitle("PCA of Fire Video Analysis frames 2118-2358. Associated Percent Variance: ")
plt.title(eig2, fontsize=8)
plt.savefig("PCA_of_Fire_Vid_Analysis2118-2358-2.png")
#plt.close()
plt.figure()
plt.streamplot(x,y,UU3,VV3,cmap='nipy_spectral')
plt.suptitle("PCA of Fire Video Analysis frames 2118-2358. Associated Percent Variance: ")
plt.title(eig3, fontsize = 8)
plt.savefig("PCA_of_Fire_Vid_Analysis2118-2358-3.png")
#plt.close()
plt.figure()
plt.streamplot(x,y,UU4,VV4,cmap='nipy_spectral')
plt.suptitle("PCA of Fire Video Analysis frames 2118-2358. Associated Percent Variance: ")
plt.title(eig4, fontsize = 8)
plt.savefig("PCA_of_Fire_Vid_Analysis2118-2358-4.png")
#plt.close()
		
