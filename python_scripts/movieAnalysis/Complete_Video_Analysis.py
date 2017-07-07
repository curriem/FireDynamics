
###############################################################################
###############################################################################
###      A Comprehensive Video Analysis Program Designed Specifically       ###
###     for analyzing smoke and fire fluid dynamics. Included are           ###
###     particle imaging velocimetry, principal component                   ###
###     analysis, and graphing programs.                                    ###
###                                                                         ###
###     Compiled and created by Elizabeth Tilly and Scott Goodrick          ###
###     with extensive but unrecorded help from other internet sources.     ###
###     June-July 2016                                                      ###
###     Computer running the program must have matplotlib,openpiv,cv2,      ###
###     numpy,copy,pylab,glob and eofs istalled for this to work.           ###
###############################################################################
###############################################################################


import matplotlib
import openpiv.tools
import openpiv.process
import openpiv.scaling
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2
import numpy as np
import copy
import pylab
import glob
import re
from eofs.standard import Eof



def pivprocess(filename,colorcode,stopframe,pixperm,fps,numpysaveto,graphsaveto,startframe=0,contouralpha=0,vertvelmin=-10,vertvelmax=10,hozvelmin=0,hozvelmax=5):
    """Completes the full PIV video processing using cv2 and OpenPiv and then graphs
    the velocity vector plot using matplotlib. Numpy is also used.

    **Arguments:**

    *filename*
            The complete name of the file from the directory where the processing program is
            found. Put as string (include quotes).
            
    *colorcode*
            0, 1, 2, or 3. This number determines the method for cutting out the green
            layer so that plant movement is not a factor in the velocity determinations.
            Specifically, it removes the green color layer.
            Option 0 leaves the frames unedited.
            Option 1 uses the average of the red and blue layers to replace the green.
            Option 2 replaces the green layer with the red layer.
            Option 3 replaces the green layer with the blue layer.
            
    *stopframe*
            The final frame to be analyzed. 
            
    *pixperm*
            Pixels per meter of the video.
            
    *fps*
            Frames per second of the video.
            
    *numpysaveto*
            File location for the numpy arrays to be stored. Example:
            'out/Vertical Velocity/arrays/another/%05d.npz'
            Must be in npz format. Include %05d to name the file by the frame because
            each vectorplot/frame will have a separate numpy array.
            If "None" is typed (with quotes), the numpy arrays will not save, however,
            since the other functions require these files, they will not work unless
            the arrays.
            
    *graphsaveto*
            File location for the graphs of the vectorplots overlaid on the frames to
            be saved. Example:
            'out/Vertical Velocity/arrays/another/%05d.png'
            Include %05d to name the file by frame because each vectorplot/frame will
            have a separate graph.

    **Optional keyword arguments:**

    *startframe*
            The first frame to be analyzed. Is automatically set to 0.
            
    *contouralpha*
            Used in the vertical velocity contourplot. Determines the transparency of
            the plot overlaid above the frame image. Values are between 0 and 1.
            Automatically set to 0 (not visible).
            
    *vertvelmin*
            The lower boundary of the range of expected vertical velocities. Is used in
            scaling the contourplot colors. Automatically set to -10. In meters/second.
            
    *vertvelmax*
            The upper boundary of the range of expected vertical velocities. Is used in
            scaling the contourplot colors. Automatically set to 10. In meters/second.
            
    *hozvelmin*
            The lower boundary of the range of expected horizontal velocities. Is used in
            scaling the streamplot colors. Automatically set at 0. In meters/second.
            
    *hozvelmax*
            The upper boundary of the range of expected horizontal velocities. Is used in
            scaling the streamplot colors. Automatically set at 5. In meters/second.
            
    **Returns:**
            A  streamplot indicating the horizontal velocities and a contourplot indicating
            vertical velocities of the fluid being analyzed of each frame overlaid on the
            frame itself. 

    **Example:**
            pivprocess('../video/tripod.mp4',1,3500,199.25,24.,'out/Vertical Velocity/arrays/another/%05d.npz',
                    'out/Vertical Velocity/arrays/another/%05d.png',startframe=0,contouralpha=0.65,vertvelmin=-8,
                    vertvelmax=8,hozvelmin=0,hozvelmax=6,xmax=500.,ymax=400.)
                    
    """



    #####Import the video#####
    vc = cv2.VideoCapture(filename)
    c=1


    ######Get frames for use in the PIV#####

    if vc.isOpened():
       rval , frame1 = vc.read()
       rval , frame2 = vc.read()
        
    else:
       rval = False



    #####Make Lists for Later#####

    U=[]
    V=[]

    #####Cuts out the green layer so that plant movement is not a factor in the velocity determinations#####	

    GreenOption = colorcode												
    if GreenOption==1: # use avg of red and blue
        frame1[:,:,1] = 0.5 * (frame1[:,:,0]+frame1[:,:,2])
        frame2[:,:,1] = 0.5 * (frame2[:,:,0]+frame2[:,:,2])
    elif GreenOption==2: #replace green with red
        frame1[:,:,1] = frame1[:,:,0]
        frame2[:,:,1] = frame2[:,:,0]
    elif GreenOption==0:
        frame1=frame1
	frame2=frame2
    else: #replace green with blue
        frame1[:,:,1] = frame1[:,:,2]
        frame2[:,:,1] = frame2[:,:,2]

    #####Starts with horizontal components calculation#####
    while rval:
        rval, frame3 = vc.read()
        if startframe < c < stopframe:												
            myimage = frame3.copy()
            
            if GreenOption==1: # use avg of red and blue
                frame3[:,:,1] = 0.5 * (frame3[:,:,0]+frame3[:,:,2])
            elif GreenOption==2: #replace green with red
                frame3[:,:,1] = frame3[:,:,0]
            elif GreenOption==0:
                frame3=frame3
            else: #replace green with blue
                frame3[:,:,1] = frame3[:,:,2]

            f1 = frame1.mean(axis=2)
            f2 = frame2.mean(axis=2)
            f3 = frame3.mean(axis=2)

            vold = (f2-f1) * (f2-f1) / (f2+f1+1)
            vnew = (f3-f2) * (f3-f2) / (f3+f2+1)
	    
            vold = 255.*(vold - vold.min() ) / (vold.max()-vold.min()+1)
            vnew = 255.*(vnew - vnew.min() ) / (vnew.max()-vnew.min()+1)

            oldimg = (255*vold).astype('int32')
            newimg = (255*vnew).astype('int32')

            u, v, sig2noise = openpiv.process.extended_search_area_piv( oldimg, newimg, window_size=24, overlap=12, dt=1./fps, search_area_size=64, sig2noise_method='peak2peak' ) 
            x, y = openpiv.process.get_coordinates( image_size=newimg.shape, window_size=24, overlap=12 )
            u, v, mask = openpiv.validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
            u, v = openpiv.filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
            x, y, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor = pixperm )				
                    
            scalef = pixperm

            U.append(u)
            V.append(v)
            if len(U)>fps:												
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
            levels = np.arange(vertvelmin,vertvelmax+1,1)										


            plt.contourf(x*scalef,y*scalef,WW,levels=levels,alpha=contouralpha,cmap='seismic')#, norm=clr.Normalize(vmin=vertvelmin,vmax=vertvelmax))
            plt.colorbar(ticks = levels, label='Vertical Velocities (m/s)', alpha = contouralpha)
            plt.streamplot(scalef*x,scalef*y, UU, VV, color=magnitude , density=2, linewidth = 1, arrowsize=1,cmap='nipy_spectral') #, norm=clr.Normalize(vmin=hozvelmin,vmax=hozvelmax) )
            plt.colorbar(extend = 'max',label='Horizontal Velocity(m/s)')
            
            




    #####Putting the image from the video in the background (Q is there to make sure the colors are normal)#####
      #      plt.streamplot(scalef*x,scalef*y, UU, VV, color='b' , density=2, linewidth= 1, arrowsize=1)
            Q = np.ones( frame3.shape ) * 1.0
            Q[:,:,2] = myimage[:,:,0] / np.float( myimage[:,:,0].max() )
            Q[:,:,1] = myimage[:,:,1] / np.float( myimage[:,:,1].max() )
            Q[:,:,0] = myimage[:,:,2] / np.float( myimage[:,:,2].max() )
            
    #####This saves the numpy arrays and the images so that they can be analyzed later on#####
    ####This particular command saves the velocities####

            if numpysaveto != None:
                    np.savez(numpysaveto %c,x=x,y=y,UU=UU,VV=VV,WW=WW)		
            plt.imshow(Q, aspect = 'auto') 						
            plt.tight_layout()    
    ####This particular command saves the images with the vector plots and vertical velocity contours####	 
            plt.title('Frame %05d'%c)										
            plt.savefig( graphsaveto %c )							
            plt.close()
          #  break
        frame1 = frame2
        frame2 = frame3


        c += 1
        cv2.waitKey(1)
    vc.release()


def movavgtur(name,component,fps,output):
    """Takes the data from the PIV analysis and determines the one second moving
    average and the turbulence from the data.

    **Arguments:**

    *name*
            The complete name of the file from the directory where the processing program is
            found. Put as string (include quotes). Must be an npz file.Include * instead of the 
            name of the individual file.
            
    *component*
            Which velocity component is being processed? UU,VV,WW are the only options. This
            can also be used with other data types however, the name of the numpy array that
            the data is saved to is required and is what would be input instead. Put in quotes.
           
    *fps*
            Frames per second of the video.

    *output*
            File location for the numpy arrays to be saved. Put in quotes. Example:
            'out/Vertical Velocity/arrays/another/%05d.png'
            Include %05d to name the file by frame because each frame will give a different set
            of numpy arrays. Must be saved as npz.
            

    **Example:**
            movavgtur('../out/*.npz', UU, 24., '../out/mvavgtur%05d.npz')
                    
    """

    #####make the dictionaries to store the abundance of velocity arrays#####

    arrays = {}

    #####put the velocity arrays in their appropriate dictionaries#####
    print name
    for np_name in glob.glob(name):
        with np.load(np_name) as data:	
            arrays[re.findall(r'\d+',np_name)[-1]] = data[component]		
	    spot = re.findall(r'\d+',np_name)[-1]


    #####finding the average#####
    ####Variables####
    window = fps/2.-1.							
    frames = arrays.keys()
    frames.sort()
    (nx,ny) = arrays[spot].shape
    print arrays[spot].shape
    nx = int(nx)
    ny = int(ny)
    print nx, ny
    nt = len(frames)						
    A = np.empty([nx,ny,nt])
    print A.shape
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
                            MA[i,j] = mov_avg(A[i,j,:], frame, int(window))
                            turb[i,j] = A[i,j,frame] - MA[i,j]
	    np.savez(output %frame, MA = MA, turb = turb)	




def graphit(name,component,destination, x=5,y=5, gx=None,gy=None,gtitle=None):
    """Graphs the change with respect to time of the chosen component at a single point.

    **Arguments:**

    *name*
            The complete name of the file from the directory where the processing program is
            found. Put as string (include quotes). Must be an npz file. Include * instead of the 
            name of the individual file name.
            
    *component*
            Which component is being processed? UU,VV,WW,MA and turb are the only options. This
            can also be used with other data types; however, the name of the numpy array that
            the data is saved to is required and is what would be input instead. Put in quotes.
           
    *destination*
            File location for the numpy arrays to be saved. Put in quotes. Example:
            'out/Vertical Velocity/arrays/another/graph.png'
             

    **Optional keyword arguments:**

    *x*
	    The x coordinate to be analyzed for the moving average and turbulence. Automatically
	    set at 5.
  
    *y*
	    The y coordinate to be analyzed for the moving average and turbulence. Automatically
	    set at 5.

    *gx*
            The xlabel for the graph. Put in quotes. Automatically set to None.
            
    *gy*
            The ylabel for the graph. Put in quotes. Automatically set to None.
            
    *gtitle*
            The title for the graph. Put in quotes. Automatically set to None.

    **Example:**
            graphit('../out/*.npz','UU','../out/mvavgtur.png',gx='Frame',gy='Velocity (m/s)',
            gtitle='X component of Velocity')
                    
    """

    #####A function that takes the arrays from the dictionaries and puts them into lists and then graphs them#####
    
    #####Make dictionaries#####
    UU = {}
                    
    #####Make Lists#####
    lUU = []
                  

    #####Put the data arrays into their respective dictionaries#####
    for np_name in glob.glob(name):
	with np.load(np_name) as data:				
            UU[re.findall(r'\d+',np_name)[-1]] = data[component]		

                   
    #####Makes and saves graph. There is no labelling.#####
    frames = UU.keys()
    frames.sort()
    for i in frames:
	a = UU[i][x,y]
	lUU.append(a)
    graph = plt.plot(lUU)
    plt.title(gtitle)
    plt.xlabel(gx)
    plt.ylabel(gy)

    plt.savefig(destination)
    plt.close()

def PCAcovariance_Analyze(name,framestart,framestop,destination,numberpca=None):
    """Completes unweighted and unscaled Principle Component Analysis on data. Specifically using covariance
    between the data sets.

    **Arguments:**

    *name*
            The complete name of the numpy array file from the directory where the processing program is
            found. Put as string (include quotes). Must be an npz file. Specifically, this is the numpy
            array file that has the UU,VV,WW data in it. Instead of putting the name of the numpy file
            (since there are a large number of them) input an asterisk (*).
            
    *framestart*
            The first frame number in the sequence of frames to be analyzed.

    *framestop*
            The last frame number in the sequence of frames to be analyzed.

    
    *destination*
            File location for the graph to be saved. Put in quotes. Example:
            'out/Vertical Velocity/arrays/another/graph.png'
            
    
    **Optional keyword arguments:**

    *numberpca*
            Number of valid eigenvalues/PCAs to be calculated. Automatically set to determine all of them.
            
   **Example:**
            PCAcovariance_Analyze('../out/velocity.npz',0,5,'../out/mvavgtur.png',numberpca=4)
                    
    """
    #####Creates Lists and Dictionaries to be used later#####
    UU = {}
    lUU = []
    VV = {}
    lVV = []

    #####Extracts numpy array data and puts it into the dictionaries for use in analysis#####
    for np_name in glob.glob(name):
            with np.load(np_name) as data:
                    UU[re.findall(r'\d+',np_name)[-1]] = data['UU']
                    VV[re.findall(r'\d+',np_name)[-1]] = data['VV']
                    
                    
    #####Takes the data from the dictionaries, sorts them, and then puts them into lists. Then turns the list into a numpy array.#####
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

    #####Puts the U and V components into one complex array with the U as the real component and the V as the imaginary#####
    velgrid = luu + (1.j * lvv)

    #####PCA#####
    solver = Eof(velgrid[framestart:framestop,:,:])
    pca = solver.eofsAsCovariance(neofs=numberpca)
    eigen = solver.eigenvalues(neigs=numberpca)

    pca = np.array(pca)
    eigen = np.array([eigen])
    intermed = eigen[0].shape
    length = intermed[0]
    print length
    #####Graphs each PCA#####
    c=0
    for i in range(length):
         UU = pca.real[i,:,:] 
         VV = pca.imag[i,:,:] 
         eig = np.array_str(eigen[0][i]) 
         (a,b) = pca[0].shape
         y,x = np.mgrid[0:a,0:b]
         plt.figure()
         plt.streamplot(x,y,UU*-1,VV*-1,cmap='nipy_spectral')
         plt.suptitle("PCA Analysis as Covariance. Associated Percent Variance: ")
         plt.title(eig, fontsize=10)
         plt.savefig(destination %i)
         plt.close()
         c+=1
    
def PCAcorrelation_Analyze(name,framestart,framestop,destination,numberpca=None):
    """Completes unweighted and unscaled Principle Component Analyzis on data. Specifically using correlation between the data
    sets.

    **Arguments:**

    *name*
            The complete name of the numpy array file from the directory where the processing program is
            found. Put as string (include quotes). Must be an npz file. Specifically, this is the numpy
            array file that has the UU,VV,WW data in it. Instead of putting the name of the numpy file
            (since there are a large number of them) input an asterisk (*).
            
    *framestart*
            The first frame number in the sequence of frames to be analyzed.

    *framestop*
            The last frame number in the sequence of frames to be analyzed.

    
    *destination*
            File location for the graph to be saved. Put in quotes. Example:
            'out/Vertical Velocity/arrays/another/graph.png'
            
    
    **Optional keyword arguments:**

    *numberpca*
            Number of valid eigenvalues/PCAs to be calculated. Automatically set to determine all of them.
            
   **Example:**
            PCAcorrelation_Analyze('../out/velocity.npz',0,5,'../out/mvavgtur.png',numberpca=4)
                    
    """
    
    #####Creates Lists and Dictionaries to be used later#####
    UU = {}
    lUU = []
    VV = {}
    lVV = []

    #####Extracts numpy array data and puts it into the dictionaries for use in analysis#####
    for np_name in glob.glob(name):
            with np.load(np_name) as data:
                    UU[re.findall(r'\d+',np_name)[-1]] = data['UU']
                    VV[re.findall(r'\d+',np_name)[-1]] = data['VV']
                    
                    
    #####Takes the data from the dictionaries, sorts them, and then puts them into lists. Then turns the list into a numpy array.#####
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

    #####Puts the U and V components into one complex array with the U as the real component and the V as the imaginary#####
    velgrid = luu + (1.j * lvv)

     #####PCA#####
    solver = Eof(velgrid[framestart:framestop,:,:])
    pca = solver.eofsAsCorrelation(neofs=numberpca)
    eigen = solver.eigenvalues(neigs=numberpca)

    pca = np.array(pca)
    eigen = np.array([eigen])
    intermed = eigen[0].shape
    length = intermed[0]
    print length
    #####Graphs each PCA#####
    c=0
    for i in range(length):
         UU = pca.real[i,:,:] 
         VV = pca.imag[i,:,:] 
         eig = np.array_str(eigen[0][i]) 
         (a,b) = pca[0].shape
         y,x = np.mgrid[0:a,0:b]
         plt.figure()
         plt.streamplot(x,y,UU*-1,VV*-1,cmap='nipy_spectral')
         plt.suptitle("PCA as Correlation Analysis. Associated Percent Variance: ")
         plt.title(eig, fontsize=10)
         plt.savefig(destination %i)
         plt.close()
         c+=1
def PCA_Analyze(name,framestart,framestop,destination,numberpca=None):
    """Completes unweighted and unscaled Principle Component Analyzis on data. 

    **Arguments:**

    *name*
            The complete name of the numpy array file from the directory where the processing program is
            found. Put as string (include quotes). Must be an npz file. Specifically, this is the numpy
            array file that has the UU,VV,WW data in it. Instead of putting the name of the numpy file
            (since there are a large number of them) input an asterisk (*).
            
    *framestart*
            The first frame number in the sequence of frames to be analyzed.

    *framestop*
            The last frame number in the sequence of frames to be analyzed.

    
    *destination*
            File location for the graph to be saved. Put in quotes. Example:
            'out/Vertical Velocity/arrays/another/graph.png'
            
    
    **Optional keyword arguments:**

    *numberpca*
            Number of valid eigenvalues/PCAs to be calculated. Automatically set to determine all of them.
            
   **Example:**
            PCA_Analyze('../out/velocity.npz',0,5,'../out/mvavgtur.png',numberpca=4)
                    
    """       
     #####Creates Lists and Dictionaries to be used later#####
    UU = {}
    lUU = []
    VV = {}
    lVV = []

    #####Extracts numpy array data and puts it into the dictionaries for use in analysis#####
    for np_name in glob.glob(name):
            with np.load(np_name) as data:
                    UU[re.findall(r'\d+',np_name)[-1]] = data['UU']
                    VV[re.findall(r'\d+',np_name)[-1]] = data['VV']
                    
                    
    #####Takes the data from the dictionaries, sorts them, and then puts them into lists. Then turns the list into a numpy array.#####
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

    #####Puts the U and V components into one complex array with the U as the real component and the V as the imaginary#####
    velgrid = luu + (1.j * lvv)
    
    #####PCA#####
    solver = Eof(velgrid[framestart:framestop,:,:])
    pca = solver.eofs(neofs=numberpca)
    eigen = solver.eigenvalues(neigs=numberpca)

    pca = np.array(pca)
    eigen = np.array([eigen])
    intermed = eigen[0].shape
    length = intermed[0]
    print length
    #####Graphs each PCA#####
    c=0
    for i in range(length):
         UU = pca.real[i,:,:] 
         VV = pca.imag[i,:,:] 
         eig = np.array_str(eigen[0][i]) 
         (a,b) = pca[0].shape
         y,x = np.mgrid[0:a,0:b]
         plt.figure()
         plt.streamplot(x,y,UU*-1.,VV*-1.,cmap='nipy_spectral')
         plt.suptitle("PCA Analysis. Associated Percent Variance: ")
         plt.title(eig, fontsize=10)
         plt.savefig(destination %i)
         plt.close()
         c+=1
    
            
        
