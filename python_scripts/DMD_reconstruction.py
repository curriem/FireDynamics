import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob

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

    filePaths = ['/Users/mcurrie/FireStats/test_data.npy']
    filePaths = glob.glob('/Users/mcurrie/FireStats/data/*npy')[:1]
    for filePath in filePaths:
        # Note: filePaths must be 2d array (loot at .mat files for correct alignment)
        fireName = filePath.split('/')[-1].strip('.npy')
        print fireName
        
        data = np.load(filePath)
        t, x, y = data.shape
        data = data.reshape((t, x*y))
        data = data.T
        data = np.nan_to_num(data)
        L = np.arange(10, t/10 * 10 -10, 30)
        
        level = 0.8
        
        perm = np.random.permutation(data.shape[0])        
        
        sizecs = np.around(data.shape[0]*level)

        data_number = data.shape[1]
        dt = 0.1
        t = np.arange(0, dt*data_number + dt, dt)
        
    
        error = np.empty(L.shape)

        count = 0
        for i in L:
            sol_dmd = np.empty((i,data_number))
            dmdbasis, y0, omega = dmd_comp_cs(data[perm[0:int(sizecs)], 0:-1], data[perm[0:int(sizecs)], 1:], i, dt, data)

            for j in range(data_number):
                sol_dmd[:,j] = y0*np.exp(omega*t[j])
                
            z = np.dot(dmdbasis, sol_dmd)
            sol_dmd_full = z.real
            
            err = np.linalg.norm(sol_dmd_full - data) / np.linalg.norm(data)
            error[count] = err
            count += 1
            sol_dmd_full = sol_dmd_full.T
            temp = []
            for frame in sol_dmd_full:
                temp.append(frame.reshape(x,y))
            temp = np.array(temp)
            
            np.save('/Users/mcurrie/FireStats/DMD/%s_reducedSol_%s.npy'%  \
                    (fireName, str(i).zfill(3)), temp)
            
        np.save('/Users/mcurrie/FireStats/DMD/%s_errors.npy'%fireName, error)    

    
main()