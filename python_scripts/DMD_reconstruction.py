import numpy as np
import matplotlib.pyplot as plt
import glob


def plotFirstPOD(data, fireName, x, y):
    U,S,V = np.linalg.svd(data, full_matrices=False)
    
    print U.shape
    firstModes = U[:, :4]
    print firstModes.shape
    firstModes = firstModes.T
    temp = []
    for frame in firstModes:
        temp.append(frame.reshape(x,y))
        
    temp = np.array(temp)
    
    n = 1
    for frame in temp:
        plt.figure()
        plt.imshow(frame, interpolation='nearest', cmap='jet', vmin=-0.006, vmax = 0.006)
        plt.axis('off')
        plt.colorbar()
        plt.title('%s, POD mode %i'%(fireName, n))
        plt.savefig('/Users/mcurrie/FireStats/DMD/firstPODmodes/%s_PODmode%i.pdf'%(fireName, n), bbox_inches='tight', clobber=True)
        plt.show()
        plt.close()
        n += 1

def plotSingVals(X, fireName):
    U1, S1, Vh = np.linalg.svd(X, full_matrices=False)    
    
    singVals = np.log10(S1)
    singVals /= singVals[0]
    plt.figure()
    plt.plot(singVals)
    plt.title('Singluar Values, %s'%fireName)
    plt.xlabel('Timesteps')
    plt.ylabel('log10(singVal)')
    plt.savefig('/Users/mcurrie/FireStats/DMD/singValPlots/%s_singVals.pdf'%fireName, clobber=True)
    plt.show()
    plt.close()
    
def dmd_comp_cs(X, X1, r, dt, oi):
    


    U1, S1, Vh = np.linalg.svd(X, full_matrices=False)
        
    S1 = np.diag(S1)
    V1 = Vh.T
    
    U = U1[:,0:r]
    S = S1[0:r, 0:r]
    V = V1[:,0:r]
    
    
    
    Sinv = np.linalg.inv(S)
    
    Atilde = np.dot(U.conj().T, np.dot(X1, np.dot(V, Sinv)))
    
    lambdaDMD, psiDMD = np.linalg.eig(Atilde)
    lambdaDMD = np.diag(lambdaDMD)
    dmdbasis = np.dot(oi[:, 1:], np.dot(V, np.dot(Sinv, psiDMD)))
    
    mu = np.diag(lambdaDMD)

    omega = np.log(mu)/dt
    
    y0 = np.linalg.lstsq(dmdbasis, oi[:,0])[0]
    
    return dmdbasis, y0, omega

def main():

    plotSV = False
    plotPOD = False

    
    filePaths = glob.glob('/Users/mcurrie/FireStats/data/s*npy')
    for filePath in filePaths:
        # Note: filePaths must be 2d array (loot at .mat files for correct alignment)
        fireName = filePath.split('/')[-1].strip('.npy')
        print fireName
        
        data = np.load(filePath)
        numTS, x, y = data.shape
        data = data.reshape((numTS, x*y))
        data = data.T
        data = np.nan_to_num(data)
        L = np.arange(10, numTS/10 * 10 -10, 10)
        L = np.array([10, 130])
        level = 0.8
        
        perm = np.random.permutation(data.shape[0])        
        
        sizecs = np.around(data.shape[0]*level)

        data_number = data.shape[1]
        dt = 0.1
        t = np.arange(0, dt*data_number + dt, dt)
        
    
        error = np.empty(L.shape)

        count = 0
        for i in L:
            try:
                sol_dmd = np.empty((i,data_number))
                dmdbasis, y0, omega = dmd_comp_cs(data[perm[0:int(sizecs)], 0:-1], data[perm[0:int(sizecs)], 1:], i, dt, data)
    
                for j in range(data_number):
                    sol_dmd[:,j] = y0*np.exp(omega*t[j])
                    
                    
                dmdBasisFrames = dmdbasis.T
                temp = []
                for frame in dmdBasisFrames:
                    temp.append(frame.reshape(x,y))
                
                dmdBasisFrames = np.array(temp)
                dmdBasisFrames = np.real(dmdBasisFrames)
                n =1
                for frame in dmdBasisFrames[:4]:
                    plt.figure()
                    plt.imshow(frame, interpolation='nearest', cmap='jet')
                    plt.axis('off')
                    plt.colorbar()
                    plt.title('%s, rank  %i, basis %i'%(fireName, i, n))
                    plt.savefig('/Users/mcurrie/FireStats/DMD/dmdBasisPlots/%s_rank%i_basisNum%i.pdf'%(fireName, i,n), bbox_inches='tight', clobber=True)
                    plt.show()
                    plt.close()
                    n += 1
            except ValueError:
                i = round(numTS - 10, -1)
                i = int(i)
                
                print 'The data does not have enough timesteps to support a rank 130 calculation. Defaulting to rank %i'%i
                sol_dmd = np.empty((i,data_number))
                dmdbasis, y0, omega = dmd_comp_cs(data[perm[0:int(sizecs)], 0:-1], data[perm[0:int(sizecs)], 1:], i, dt, data)

                for j in range(data_number):
                    sol_dmd[:,j] = y0*np.exp(omega*t[j])
                
                
                dmdBasisFrames = dmdbasis.T
                temp = []
                for frame in dmdBasisFrames:
                    temp.append(frame.reshape(x,y))
            
                dmdBasisFrames = np.array(temp)
                dmdBasisFrames = np.real(dmdBasisFrames)
                n =1
                for frame in dmdBasisFrames[:4]:
                    plt.figure()
                    plt.imshow(frame, interpolation='nearest', cmap='jet')
                    plt.axis('off')
                    plt.colorbar()
                    plt.title('%s, rank  %i, basis %i'%(fireName, i, n))
                    plt.savefig('/Users/mcurrie/FireStats/DMD/dmdBasisPlots/%s_rank%i_basisNum%i.pdf'%(fireName, i,n), bbox_inches='tight', clobber=True)
                    plt.show()
                    plt.close()
                    n += 1
#==============================================================================
#             z = np.dot(dmdbasis, sol_dmd)
#             sol_dmd_full = z.real
#             
#             err = np.linalg.norm(sol_dmd_full - data) / np.linalg.norm(data)
#             error[count] = err
#             count += 1
#             sol_dmd_full = sol_dmd_full.T
#             temp = []
#             for frame in sol_dmd_full:
#                 temp.append(frame.reshape(x,y))
#             temp = np.array(temp)
#             
#             np.save('/Users/mcurrie/FireStats/DMD/%s_reducedSol_%s.npy'%  \
#                     (fireName, str(i).zfill(3)), temp)
#             
#         np.save('/Users/mcurrie/FireStats/DMD/%s_errors.npy'%fireName, error)    
#         
#==============================================================================
        if plotSV:
            X = data[perm[0:int(sizecs)], 0:-1]
            plotSingVals(X, fireName)
        
        if plotPOD:
            plotFirstPOD(data, fireName, x, y)
    
main()