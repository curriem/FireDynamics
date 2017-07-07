#####Makes graphs of the velocities, their one second moving averages, and their turbulence components.#####
####If this is to be used on another computer or for other videos and arrays all of the file names will have to be changed####

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob


#####Make dictionaries#####
UU = {}
VV = {}
WW = {}
maU = {}
maV = {}
maW = {}
tU = {}
tV = {}
tW = {}

#####Make Lists#####
lUU = []
lVV = []
lWW = []
lmaU = []
lmaV = []
lmaW = []
ltU = []
ltV = []
ltW = []




#####Put the data arrays into their respective dictionaries#####
for np_name in glob.glob('out/Vertical Velocity/arrays/*.npz'):
	with np.load(np_name) as data:
		UU[np_name[29:34]] = data['UU']
		VV[np_name[29:34]] = data['VV']
		WW[np_name[29:34]] = data['WW']
		
#for np_name in glob.glob('out/Vertical Velocity/mvavgturb/UU/*.npz'):
#	with np.load(np_name) as data:
#		maU[np_name[35:40]] = data['MA']
#		tU[np_name[35:40]] = data['turb']

#for np_name in glob.glob('out/Vertical Velocity/mvavgturb/VV/*.npz'):
#	with np.load(np_name) as data:
#		maV[np_name[35:40]] = data['MA']
#		tV[np_name[35:40]] = data['turb']

#for np_name in glob.glob('out/Vertical Velocity/mvavgturb/WW/*.npz'):
#	with np.load(np_name) as data:
#		maW[np_name[35:40]] = data['MA']
#		tW[np_name[35:40]] = data['turb']




#####A function that takes the arrays from the dictionaries and puts them into lists and then graphs them#####
def graphme(dictionary, mylist):
	frames = dictionary.keys()
	frames.sort()
	for i in frames:
		a = dictionary[i][5,50]
		mylist.append(a)
	return plt.plot(mylist)
#####Makes and saves each graph. There is no labelling.#####
Xvel = graphme(UU,lUU)
plt.xlabel('Frame')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity in X direction')
plt.savefig('out/X_velocity_component.png')
#plt.show()
plt.close()
Yvel = graphme(VV,lVV)
plt.xlabel('Frame')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity in Y direction')
plt.savefig('out/Y_velocity_component.png')
#plt.show()
plt.close()
Zvel = graphme(WW,lWW)
plt.xlabel('Frame')
plt.ylabel('Velocity (m/s)')
plt.title('Velocity in Z direction')
plt.savefig('out/Z_velocity_component.png')
#plt.show()
plt.close()
#Xavg = graphme(maU,lmaU)
#plt.xlabel('Frame')
#plt.ylabel('Velocity (m/s)')
#plt.title('Moving Average of Velocity in X direction')
#plt.savefig('out/Moving_Average_X.png')
#plt.show()
#plt.close()
#Yavg = graphme(maV,lmaV)
#plt.xlabel('Frame')
#plt.ylabel('Velocity (m/s)')
#plt.title('Moving Average of Velocity in Y direction')
#plt.savefig('out/Moving_Average_Y.png')
#plt.show()
#plt.close()
#Zavg = graphme(maW,lmaW)
#plt.xlabel('Frame')
#plt.ylabel('Velocity (m/s)')
#plt.title('Moving Average of Velocity in Z direction')
#plt.savefig('out/Moving_Average_Z.png')
#plt.show()
#plt.close()
#Xtur = graphme(tU,ltU)
#plt.xlabel('Frame')
#plt.ylabel('Turbulence')
#plt.title('Turbulence in X direction')
#plt.savefig('out/Turbulence_in_X.png')
##plt.show()
#plt.close()
#Ytur = graphme(tV,ltV)
#plt.xlabel('Frame')
#plt.ylabel('Turbulence')
#plt.title('Turbulence in Y direction')
#plt.savefig('out/Turbulence_in_Y.png')
#plt.show()
#plt.close()
#Ztur = graphme(tW,ltW)
#plt.xlabel('Frame')
#plt.ylabel('Turbulence')
#plt.title('Turbulence in Z direction')
#plt.savefig('out/Turbulence_in_Z.png')
##plt.show()
#plt.close()

