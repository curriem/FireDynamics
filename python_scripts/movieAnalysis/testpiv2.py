#####Determines the horizontal and vertical velocities of smoke using particle image velocimetry (PIV)#####
####Things to change if this file is to be used for a different video or a different computer are marked with #*#. ####
import matplotlib
from tempfile import TemporaryFile
import openpiv.tools
import openpiv.process
import openpiv.scaling
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2
import numpy as np
import copy


#####Import the video#####
vc = cv2.VideoCapture('/Users/mcurrie/FireDynamics/data/f3/out.avi') #'../video/tripod.mp4') #fireflux.flv')							#*#
c=1


######Get frames for use in the PIV#####

if vc.isOpened():
    rval , frame1 = vc.read()
    rval , frame2 = vc.read()
    
else:
    rval = False

print rval

#####Make Lists for Later#####

U=[]
V=[]

#####Cuts out the green layer so that plant movement is not a factor in the velocity determinations#####	#*#This section may or may not need to be changed depending on the noise gotten from foliage

##GreenOption = 1 # 1 or 2 or 3												
##if GreenOption==1: # use avg of red and blue
##    frame1[:,:,1] = 0.5 * (frame1[:,:,0]+frame1[:,:,2])
##    frame2[:,:,1] = 0.5 * (frame2[:,:,0]+frame2[:,:,2])
##elif GreenOption==2: #replace green with red
##    frame1[:,:,1] = frame1[:,:,0]
##    frame2[:,:,1] = frame2[:,:,0]
##else: #replace green with blue
##    frame1[:,:,1] = frame1[:,:,2]
##    frame2[:,:,1] = frame2[:,:,2]

#####Starts with horizontal components calculation#####
while rval:
    rval, frame3 = vc.read()
    if 160 < c < 3500:												#*#Input number of total frames
	myimage = frame3.copy()
##	
##	if GreenOption==1: # use avg of red and blue
##            frame3[:,:,1] = 0.5 * (frame3[:,:,0]+frame3[:,:,2])
##        elif GreenOption==2: #replace green with red
##            frame3[:,:,1] = frame3[:,:,0]
##        else: #replace green with blue
##            frame3[:,:,1] = frame3[:,:,2]

        f1 = frame1.mean(axis=2)#[5:225,200:640]
        f2 = frame2.mean(axis=2)#[5:225,200:640]
        f3 = frame3.mean(axis=2)#[5:225,200:640]

        vold = (f2-f1) * (f2-f1) / (f2+f1+1)
        vnew = (f3-f2) * (f3-f2) / (f3+f2+1)
        print vold.max(),vold.min(), vnew.max(),vnew.min()
        vold = 255.*(vold - vold.min() ) / (vold.max()-vold.min()+1)
        vnew = 255.*(vnew - vnew.min() ) / (vnew.max()-vnew.min()+1)

        oldimg = (255*vold).astype('int32')
        newimg = (255*vnew).astype('int32')

        u, v, sig2noise = openpiv.process.extended_search_area_piv( oldimg, newimg, window_size=24, overlap=12, dt=1./24., search_area_size=64, sig2noise_method='peak2peak' ) #*#dt involves fps
        x, y = openpiv.process.get_coordinates( image_size=newimg.shape, window_size=24, overlap=12 )
        u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
        u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
        x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = 3086. )				#*#For scaling factor see note below
		
	

        U.append(u)
        V.append(v)
        if len(U)>24:												#*#Involves fps
            junk = U.pop(0)
            junk = V.pop(0)
        for ui in U:
            if len(U)==1:
                UU=ui
            else:
                UU=UU+ui
        UU = UU / float(len(U))
        for vi in V:
            if len(V)==1:
                VV=vi
            else:
                VV=VV+vi
        VV = VV / float(len(V))
        magnitude = np.sqrt( UU*UU+VV*VV )
        

######Vertical component calculations#####
        dvdy,dvdx = np.gradient( VV )
        dudy,dudx = np.gradient( UU )
        Vort = dvdx-dudy
	divergence = dudx+dvdy        
	WW = -2*divergence	


#####Making the plots#####
	plt.figure()
##	levels = np.arange(-8,9,1)										#*#Expected vertical velocity range


##	plt.contourf(x*40.,y*40.,WW,levels=levels,alpha=0.65,cmap='seismic', norm=clr.Normalize(vmin=-8,vmax=8))#*#Normalize for vertical velocity range
##	plt.colorbar(ticks = levels, label='Vertical Velocities (m/s)')
#*#when scaling the velocities to the video, you have to change three things: the multiplier on the x and y in the contourplot (right above) and the streamplot (right below) and the scaling factor above. You have to scale the three numbers by the same amount, so if you change the scaling factor by a factor of 10, you also have to change the multiplier by a factor of 10#*#
      
	plt.streamplot(3086.*x,3086.*y, UU, VV, color=magnitude , density=2, linewidth = 1, arrowsize=1,cmap='nipy_spectral')#, norm=clr.Normalize(vmin=0,vmax=6.00) )#*#Normalize for horizontal velocity range
	
	plt.colorbar(ticks=[0.,.50,1.00,1.50,2.00,2.50,3.00,3.50,4.00,4.50,5.00,5.50,6.,6.5], extend = 'max',label='Horizontal Velocity(m/s)')#*#Ticks for horizontal velocity range
	
	




#####Putting the image from the video in the background (Q is there to make sure the colors are normal)#####
	#plt.streamplot(3086.*x,3086.*y, UU, VV, color='b' , density=2, linewidth= 1, arrowsize=1)
        Q = np.ones( frame3.shape ) * 1.0
        Q[:,:,2] = frame3[:,:,0] / np.float( frame3[:,:,0].max() )
        Q[:,:,1] = frame3[:,:,1] / np.float( frame3[:,:,1].max() )
        Q[:,:,0] = frame3[:,:,2] / np.float( frame3[:,:,2].max() )
        #print Q.shape, Q[:,:,0].max(), Q[:,:,0].min(), Q[:,:,1].max(), Q[:,:,1].min(), Q[:,:,2].max(), Q[:,:,2].min()
#####This saves the numpy arrays and the images so that they can be analyzed later on#####
####This particular command saves the velocities####
	np.savez('out/Vertical Velocity/arrays/test-%05d.npz' %c,x=x,y=y,UU=UU,VV=VV,WW=WW)			#*#
	plt.imshow(Q, aspect = 'auto', extent=[0.0,720.,0.0,500.],alpha = 0.7) 						#*#
	#plt.tight_layout()    
####This particular command saves the images with the vector plots and vertical velocity contours####	 
	plt.title('Frame %05d'%c)										#*#
        plt.savefig( 'out/Vertical Velocity/test-%05d.png' %c )							#*#
        plt.close()
   #     break
    frame1 = frame2
    frame2 = frame3

## The video taken (which is being analyzed) is approximately 5.069 m by 9.054 m which is 199.25 pixels per meter. ## 

    c += 1
    cv2.waitKey(1)
vc.release()
