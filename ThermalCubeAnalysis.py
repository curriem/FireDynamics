import numpy as np

from matplotlib import pyplot as plt

import os





"""

Channel: Ch1   | Ch2   | Ch3   | Ch4    | Ch5    | Ch6   | Ch7   | Ch8   | Ch9   | Ch10  

Units:   deg C | deg C | deg C | deg C  | Volts  | Volts | Volts | Volts | Volts | Volts 

Desc:    front | mid   | rear  | sensor | supply | batt  | O2    | CO2   | CO    | %RH 

column:  3     | 4     | 5     | 6      | 7      | 8     | 9     | 10    | 11    | 12

"""

def openFile(filePath):
    data = np.genfromtxt(filePath, delimiter=',', skip_header=48)
    return data

def Temperature(data):

    """Gets the front and rear thermal cube temp from the CSV file

    Args:

        String containing filename of CSV file located on Windows desktop

    Returns:

        Three floats - the time series of front and rear temp and the 

        corresponding time vector

    """

    

    

    time = data[:, 0]    

    T1 = data[:, 3]

    T3 = data[:, 5]

    return time, T1, T3

        

def HeatFlux(data):

    """Gets the front and rear thermal cube temp from the CSV file 

       Calculates the heat flux, both uncompensated and accounting for radiative 

       losses 

    Args:

        String containing filename of CSV file located on Windows desktop

    Returns:

        Three floats - the time series of the compensated and uncompensated 

        heat flux and the corresponding time vector

    """

    

    time = data[:, 0] 

    T1 = data[:, 3] # Front surface temp

    T3 = data[:, 5] # Back surface temp

    T4 = data[:, 6] # Ambient air temp near sensor



    # k, thermal conductivity (W m-1 deg C)

    # L, Thermal cube width, meters

    # d, Distance to back temp probe, meters

    # sigma, Stefan-Boltzmann constant, W/(m2K4)

    # epsilon, emmissivity

    k, L, d, sigma, epsilon  = 200., 25.4E-3, 12.7E-3, 5.67E-08, 0.95



    Q1 = k*(T1 - T3)/(L - d)/1000

    

    # with radiative losses

    Q2 = (1/epsilon/1000)*(Q1 + sigma*epsilon*(np.power(T3, 4) + 

          np.power(T1, 4) - 2*np.power(T4, 4)))



    return time, Q1, Q2



def RelativeHumidity(str):

    """Gets the sensor voltage, battery voltage, and sensor temperature from 

       the CSV file 

       Calculate the temperature compensated relative humdity

    Args:

        String containing filename of CSV file located on Windows desktop

    Returns:

        Two floats, the time series of temp-compensated relative humidity and

        the time vector

    """

    f = os.path.join(r'C:\Users\Administrator\Desktop', str)

    data = np.genfromtxt(f, delimiter=',')



    time = data[2:, 0] 

    V5 = data[2:, 7]  # Channel 5 - Sensor supply voltage

    V10 = data[2:, 12] # Channel 10 - RH sensor output voltage

    T4 = data[2:, 6] # Ambient air temp

    

    # b, y intercept of constitutive equation (volts)  

    # m, slope of constitutive equation  (V/%RH)

    

    # uncompensated %RH

    b = 0.16*V5

    m = 0.0062*V5

    RHu = (V10 - b) / m

    

    # temp compenstated %RH

    RH = RHu / (1.0546 - 0.00216 * T4)

    

    return time, RH

    



def Oxygen(str):

    """Gets the sensor voltage of the O2 sensor from the CSV file 

       Calculate the oxygen concentration in %

    Args:

        String containing filename of CSV file located on Windows desktop

    Returns:

        Two floats, the time series of oxygen conc. and the time vector

    """

    

    f = os.path.join(r'C:\Users\Administrator\Desktop', str)

    data = np.genfromtxt(f, delimiter=',')



    time = data[2:, 0] 

    V10 = data[2:, 9] # Channel 7 - O2 sensor output voltage



    O2 = np.power(10, 4) * V10 / (121 * 7.43)

   

    return time, O2

    

    


filePath = '/Users/mcurrie/FireDynamics/data/FireBox/170421-133855_UG.CSV'

data = openFile(filePath)
savePath = '/Users/mcurrie/FireDynamics/data/FireBox/'

############################ Get the data ####################################

time1, T1, T3 = Temperature(data)


time2, Q1, Q2 = HeatFlux(data)



#time3, RH     = RelativeHumidity('test.csv')

#time4, O2     = Oxygen('test.csv')



#################### Save plots of the data to file ##########################



######################### Temperature plots ##################################

fig1 = plt.figure()

plt.plot(time1, T1, color='r', linewidth=1)

plt.plot(time1, T3, color='b', linewidth=1)



plt.title('Temperature')

plt.ylabel('Temperature, deg C')

plt.xlabel('Time, sec')

    

plt.savefig(savePath+'TempFig2.pdf',dpi=None, facecolor='w', 

            edgecolor='w', orientation='portrait', papertype=None, format=None,

            transparent=False, bbox_inches=None, pad_inches=0.1,

            frameon=None)    


######################## Heat flux plots #####################################

fig2 = plt.figure()    

plt.plot(time2, Q1, color='r', linewidth=1)

#plt.plot(time2, Q2, color='b', linewidth=1)



plt.title('Heat Flux')

plt.ylabel(r'Heat flux, W $\mathregular{m^{-2}}$')

plt.xlabel('Time, sec')


plt.savefig(savePath+'FluxFig2.pdf',dpi=None, facecolor='w', 

            edgecolor='w', orientation='portrait', papertype=None, format=None,

            transparent=False, bbox_inches=None, pad_inches=0.1,

            frameon=None)

    

#################### Relative humidity plots ################################
"""
fig3 = plt.figure()    

plt.plot(time3, RH, color='r', linewidth=1)



plt.title('% Relative Humidity')

plt.ylabel(r'Relative Humidity, %')

plt.xlabel('Time, sec')

    

plt.savefig(r'C:\Users\Administrator\Desktop\RHFig.pdf',dpi=None, facecolor='w', 

            edgecolor='w', orientation='portrait', papertype=None, format=None,

            transparent=False, bbox_inches=None, pad_inches=0.1,

            frameon=None)

            

####################### O2 concentration plots ################################

fig4 = plt.figure()    

plt.plot(time3, O2, color='r', linewidth=1)



plt.title('Oxygen Conc.')

plt.ylabel(r'O$\mathregular{_{2}}$, %')

plt.xlabel('Time, sec')

    

plt.savefig(r'C:\Users\Administrator\Desktop\O2Fig.pdf',dpi=None, facecolor='w', 

            edgecolor='w', orientation='portrait', papertype=None, format=None,

            transparent=False, bbox_inches=None, pad_inches=0.1,

            frameon=None)

"""