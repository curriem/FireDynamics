#####A program to calculate the one second moving averages and turbulence factors of the velocimetry graphs from testpiv2.py#####
#####If this program is to be used for other arrays, computers, or videos, things that need to change are marked by #*#. #####
import numpy as np
import glob
import pylab

#####make the dictionaries to store the abundance of velocity arrays#####

arrays = {}

#####put the velocity arrays in their appropriate dictionaries#####

for np_name in glob.glob('out/Vertical Velocity/arrays/*.npz'): #*#
	with np.load(np_name) as data:				#*#
		arrays[np_name[29:34]] = data['VV']		#*#



#####finding the average#####
####Variables####
window = 11							#*# This is equal to the fps/2 - 1
frames = arrays.keys()
frames.sort()
nx = 89								#*# X dimension of the video
ny = 159							#*# Y dimension of the video
nt = len(frames)						#*# The number of total frames
A = np.empty([nx,ny,nt])
MA = np.empty([nx,ny])
turb = np.empty([nx,ny])
count = 0

####Moving Average Function####
def mov_avg(x,i,w):
    inp = list(x)
    out = list(x)
    start = max(0,i-w)
    end = min(len(inp), i+w)
    total = sum( inp[start:end] )
    count = float( end-start+1)
    out = total/count
    return out

####Puts the velocity arrays into a large numpy array in frame order####
for key in frames:
	A[:,:,count] = arrays[key]
	count += 1
####Calculates the moving average and the turbulence and then saves the arrays####
for frame in range(nt):
	for i in range(nx):
		for j in range(ny):
			MA[i,j] = mov_avg(A[i,j,:], frame, window)
			turb[i,j] = A[i,j,frame] - MA[i,j]
	np.savez('out/Vertical Velocity/mvavgturb/VV/%05d.npz' %frame, MA = MA, turb = turb)	#*# 



