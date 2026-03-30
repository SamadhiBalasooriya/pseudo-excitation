from matrix import*  
from solver import Newmarksuper_HSI2,accdyn_super 
from pedestrian import* 
from matplotlib import pyplot as plt
import timeit
import numpy as np
from scipy.stats import norm
from ZIV_SM import compute_subharmonic_dlf, compute_dlf

 

""" Data from the table for Harmonics Zivanovic and Pavic (2007)"""
i_values = [1, 2, 3, 4, 5]
a_i1_values = [0.785200, 0.513000, 0.390800, 0.325500, 0.280600]
b_i1_values = [0.999900, 2.000000, 3.000000, 4.000000, 4.999000]
c_i1_values = [0.008314, 0.011050, 0.009560, 0.008797, 0.007939]

a_i2_values = [0.020600, 0.133000, 0.156700, 0.164700, 0.158400]
b_i2_values = [1.034000, 1.957000, 3.000000, 4.001000, 5.004000]
c_i2_values = [0.252400, 0.263200, 0.055250, 0.066410, 0.078250]

a_i3_values = [0.107400, -0.049840, 0.068660, 0.068880, 0.072890]
b_i3_values = [1.001000, 1.882000, 2.957000, 3.991000, 4.987000]
c_i3_values = [0.036530, 0.058070, 0.560700, 0.375000, 0.450100]



""" Data from the table for Sub Harmonics Zivanovic and Pavic (2007)"""
a_s_i1_values = [0.340600, 0.302400, 0.262700, 0.234400, 0.264500]
b_s_i1_values = [0.498800, 1.500000, 2.500000, 3.501000, 4.499000]
c_s_i1_values = [0.008337, 0.008735, 0.009748, 0.009898, 0.010190]
a_s_i2_values = [0.280300, 0.134500, 0.245600, 0.235500, 0.238900]
b_s_i2_values = [1.133000, 1.532000, 0.231200, -1.576000, 1.153000]
c_s_i2_values = [0.638800, 0.723300, 2.932000, 7.050000, 4.561000]








t = np.array(np.arange(0, (50+1) / 1.25, 0.01)) 
no_simulations = 20
all_forces = np.zeros((no_simulations, np.size(t)))


for i in range(no_simulations):
    
    # probabilitstic parameters
    mean_pace = 2 #Hz  2005 pachi
    pace_COV = 0.1
    pace = np.random.normal(mean_pace,pace_COV*mean_pace)

    mean_DLF1 = -0.2649*pace**3 +1.3206*pace**2-1.7597*pace+0.7613 #formular from papere pg3


    x1 = np.arange(0, 2, 0.01) #multilpication factor transforming mean DLF into the distribution of DLF
    pdf = norm.pdf(x1,1, 0.16)
    #DLF1 = mean_DLF1*pdf
    #plt.plot(x1, pdf)



    DLF2 = 0.07
    DLF3 = 0.05
    DLF4 = 0.05
    DLF5 = 0.03 
    DLF=np.array([mean_DLF1, DLF2, DLF3, DLF4, DLF5])

    STD_DLF2 = 0.03
    STD_DLF3 = 0.02
    STD_DLF4 = 0.02
    STD_DLF5 = 0.015




    DLF_s_1= 0.026*mean_DLF1 + 0.0031
    DLF_s_2= 0.074*mean_DLF1 + 0.01   
    DLF_s_3= 0.012*mean_DLF1 + 0.016
    DLF_s_4= 0.013*mean_DLF1 + 0.0093
    DLF_s_5= 0.015*mean_DLF1 + 0.0072

    DLF_s = np.array([DLF_s_1, DLF_s_2, DLF_s_3, DLF_s_4, DLF_s_5])

    f_values = [pace, 2*pace, 3*pace, 4*pace, 5*pace]
    i_values = [1,2,3,4,5]
    dlf_values = []
    x = np.linspace(0, 10, 1000)
    timedomainforce = np.zeros(t.size)

    for i in range(np.size(i_values)):
        f = f_values[i]
        f_range = np.arange(f - 0.5, f + 0.5, 0.01)
        
        dlf_values = [compute_dlf(f_val, i, pace) for f_val in f_range]
        for k in range(np.size(f_range)):
            timedomainforce += dlf_values[k]*np.sin(2*np.pi*f_range[k]*t+np.random.uniform(np.pi, 2*np.pi))*750*DLF[i]

        f_range2 = np.arange(f - 1.5, f - 0.5, 0.01)  # Adjust the step size as needed
        dlf_values2 = [compute_subharmonic_dlf(f_val, i,pace) for f_val in f_range2]

        for k in range(np.size(f_range2)):
            timedomainforce += dlf_values2[k]*np.sin(2*np.pi*f_range2[k]*t+np.random.uniform(np.pi, 2*np.pi))*750*DLF_s[i]
    all_forces[i,:]= timedomainforce


rms_footfall = np.sqrt(np.mean(np.abs(all_forces)**2, axis=0))

from scipy.signal import welch

# Compute the PSD of the sin signal
fs = 1 / (t[1] - t[0])  # Sampling frequency
frequencies, psd = welch(rms_footfall, fs=fs,nperseg=1000, nfft=2000 )
frequencies = np.array(frequencies)
psd = np.array(psd)
frequencies=frequencies[frequencies<(10)] #adjust range of frequency range suitable for analysis
psd=psd[:np.size(frequencies)]

# Plot the PSD
plt.figure()
plt.plot(frequencies, psd)
plt.title("Power Spectral Density of sin")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (V^2/Hz)")
plt.grid(True)
plt.show()