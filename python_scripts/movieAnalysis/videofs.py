#####Completes PCA of velocimetry videos#####
####Things to change if this file is to be used for a different video or a different computer are marked with #*#. ####
import numpy as np
from eofs.standard import Eof
import glob
import matplotlib.pyplot as plt

####Create dictionaries and lists for later usage####
UU = {}
lUU = []
VV = {}
lVV = []
x = {}
lx = []
y = {}
ly = []

####import, sort, and store the arrays for analysis####
for np_name in glob.glob('out/Vertical Velocity/arrays/*.npz'):           #*#
	with np.load(np_name) as data:
		UU[np_name[29:34]] = data['UU']
		VV[np_name[29:34]] = data['VV']
		
		

uframes = UU.keys()
uframes.sort()
vframes = VV.keys()
vframes.sort()




for i in uframes:
	u = UU[i]
	lUU.append(u)
	
for i in vframes:
	v = VV[i]
	lVV.append(v)

luu = np.asarray(lUU)
lvv = np.asarray(lVV)

####PCA####
velgrid = luu + (1.j * lvv)

solver = Eof(velgrid[2118:2358,:,:])                                      #*# Choose which frames
pca = solver.eofsAsCovariance(neofs = 4)                                  #*# neofs and type of eofs
eigen = solver.eigenvalues(neigs=4)                                       #*# neigs


####Save new numpy array for later graphing####

np.savez('PCA.npz', pca=pca, eigen=eigen)                                 #*#


