import math
import numpy as np
import random
from matplotlib import pyplot as plt
from pedestrian import *
from scipy.signal import welch, detrend, butter, filtfilt
from solver import* #Phi_matrix ,accdyn_super,MatrixAssemblesymetric,calculate_frf_and_accelerance,g_pj
from matrix import bridge


#step 1 setup beam and pedestrians

#beam

length = 50  # L - Length (m)
width = 2  # b - Width (m)
height = 0.6  # h - Height (m)
E = 200e9  # E - Young's modulus (N/m^2)
modalDampingRatio = 0.005  # xi - Modal damping ratio of the beam
nHigh = 3  # nHigh - Higher mode for damping matrix
beamFreq =2 #Hz
area = 0.3162  # A - Cross-section area (m^2)
linearMass = 500  # m - Linear mass (kg/m)
x_interested= length/2
numbers = 2

#ped
numped = 1
pedmass = 80     #kg
peddamp = .3    
#pedstiff = 25000 #N/m
pedpace  = 2     #Hz
pedphase = 0
pedInlocation = 0
pedvelocity = 1.25
pedBodyF= 2 #Hz

#ped
kped1=(2*np.pi*pedBodyF)**2*pedmass
cped1 = (2*np.pi*pedBodyF)*2*peddamp*pedmass

mped = np.array([pedmass])
cped = np.array([cped1])
kped = np.array([kped1])


modulus =linearMass * ((2 * math.pi * beamFreq) * (math.pi / length) ** (-2)) ** 2  #E*(width*height**3)/12

#set time info
hht=0.01

#initial possition vector.......formultiple ped all these would become matrices

#xrb=np.zeros(1,numped)
xrb=[0]

Bridge = bridge(   
    length = length,                 # m
    modulus = modulus,               # N m^2
    density = linearMass,            # kg/m
    damp    = modalDampingRatio ,    #%
    numbers =numbers  )                   #modes


N_bridge = 2




# probabilitstic parameters
mean_pace = 2 #Hz  2005 pachi
pace_COV = 0.1

mean_mass= 70 #kg
mass_COV= 0.17 #from butz 2008

mean_velocity = 1.3
std_velocity = 0.12 #pachi 2005

mean_alpha = np.array([0.41 * (mean_pace - 0.95),
                0.069 + 0.0056 * mean_pace,
                .033 + 0.0064 * mean_pace,
                0.013 + 0.0065 *mean_pace])

alpha_COV= np.array([0.17,0.4,0.4,0.4])


alpha_std = mean_alpha*alpha_COV

# Generate a random variable from a normal distribution with considered mean and std_dev from literiture

randomPace = random.gauss(mean_pace, pace_COV*mean_pace)
randomMass = random.gauss(mean_mass, mass_COV*mean_mass)
randomAlpha = [random.gauss(mean_alpha[i], alpha_std[i]) for i in range(len(alpha_COV))]
#randomPhase = [random.uniform(0, 2 * math.pi) for i in range(len(mean_alpha)+1)]
randomVelocity = random.gauss(mean_velocity,std_velocity)
randomPhase = np.zeros(5)


print(randomPace)
print(randomMass)
print(randomAlpha)
print(randomPhase)
print(randomVelocity)

t = np.array(np.arange(0, (length + 1) / randomVelocity, hht))

Human = Pedestrian(
         mass = pedmass,     #kg
         damp = peddamp ,   #%
         stiff = kped, #N/m
         pace  = pedpace ,    #Hz
         phase = pedphase,
         location = pedInlocation,
         velocity = pedvelocity,
         
        iSync=0)
'''
Human = Pedestrian(
         mass = randomMass,     #kg
         damp = peddamp ,   #%
         stiff = kped, #N/m
         pace  = randomPace ,    #Hz
         phase = randomPhase,
         location = pedInlocation,
         velocity = randomVelocity,
         
         iSync=0)'''

#ped = RandPedestrian(randomMass, randomPace, randomPhase, randomAlpha)
n=1
numped=1
#j=np.size(t)
xr=[0]
#force_at_time_t = np.zeros(np.size(t))


#modal force
F=np.zeros((N_bridge,np.size(t)))
for i in range (np.size(t)):
     
    force_at_time_t = calcDynamicPedForce(Human,t[i])         #calculates the human force at time t
    NN=Phi_matrix(xr,length,linearMass,N_bridge,numped)       #mode shape
    F[:,[i]] = NN * force_at_time_t                           #modalforce
    xr=np.add(xr,randomVelocity*hht)


plt.plot(t, F[0, :], label="mod 1", color='r')  # First row
plt.plot(t, F[1, :], label="mod 2", color='g')  # Second row
plt.show()



fs=1/hht

# Compute the FFT of the detrended signal for each row
fft_values = np.fft.fft(F, axis=1)

n = F.shape[1]  # Number of sample points per row
frequencies = np.fft.fftfreq(n, 1/fs)  # Frequency bins

# Only take the positive half of the frequencies and FFT
positive_frequencies = frequencies[:n//2]
positive_fft_values = np.abs(fft_values[:, :n//2])

# Plot the FFT for each row
plt.figure(figsize=(12, 6))
for i in range(N_bridge):
    plt.plot(positive_frequencies, positive_fft_values[i, :], label=f'Row {i+1}')

plt.title("Frequency-Domain Signal (FFT) for Each Row")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()


fs = 1/hht 
frequencies, psd_temp = welch(F[0, :], fs=fs, nperseg=1000, nfft=2000,)  # Run welch once to determine size
psd_values = np.zeros((N_bridge, len(psd_temp))) 
#psd_values = np.zeros((N_bridge,np.size(t)))

# Loop through each row of matrix F (each mode) and compute its PSD
for i in range(N_bridge):  # Loop over each row (mode)
    frequencies, psd_values[[i],:] = welch(F[i, :], fs=fs, nperseg=1000, noverlap=500 ,nfft=2000)  # Adjust nperseg as needed
    

frequencies=frequencies[frequencies<(10)] #adjust range of frequency range suitable for analysis
psd=psd_values[:,:np.size(frequencies)]   
print(np.size(frequencies))
print(np.size(psd))

# Plot the PSD for each mode
plt.figure()  
plt.plot(frequencies, psd[0,:], label="mod 1", color='r')
plt.plot(frequencies, psd[1,:], label="mod 2", color='g')
#plt.plot(frequencies, psd[2,:], label="mod 3", color='b')
#plt.plot(frequencies, psd[3,:], label="mod 4", color='orange')
plt.title("psd of modal forces")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.legend()
plt.grid()
plt.show()

M,K,C,_=MatrixAssemble(Human,Bridge,mped,kped,cped,xrb,length,linearMass,N_bridge,numped,t[2000])
frequency_range = frequencies*2*np.pi #np.arange(0.1,63,0.1)
_,accelerance = calc_frf(M,C,K,frequency_range)

n_dof = M.shape[0]
for dof in range(n_dof):
    plt.plot(frequency_range/(2*np.pi), np.abs(accelerance[dof, dof, :]), label=f'DOF {dof + 1}')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('FRF Magnitude')
    plt.title('Frequency Response Function (FRF) Magnitude')
    plt.legend()
    plt.grid(True)
plt.show()

# Plot PSD and FRF in the same window using subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot the PSD for each mode
axs[0].plot(frequencies, psd[0, :],label="mod 1", color='r' )
axs[0].plot(frequencies, psd[1, :], label="mod 2", color='g')
# axs[0].plot(frequencies, psd[2, :], label="mod 3", color='b')
# axs[0].plot(frequencies, psd[3, :], label="mod 4", color='orange')
axs[0].set_title("PSD of Modal Forces")
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Power Spectral Density')
axs[0].legend()
axs[0].grid()

# Plot FRF for each DOF
#n_dof = M.shape[0]
#for dof in range(n_dof):
axs[1].plot(frequency_range / (2 * np.pi), np.abs(accelerance[0, 0, :]), label="mod 11", color='r')
axs[1].plot(frequency_range / (2 * np.pi), np.abs(accelerance[0, 1, :]), label="mod 12", color='g')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('FRF Magnitude')
axs[1].set_title('Frequency Response Function (FRF) Magnitude')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

#plot_frf_magnitude(M, C, K, frequencies)

delta_omega=frequency_range[3]-frequency_range[2]

from scipy.integrate import quad
frequency_range =  frequencies*2*np.pi #np.arange(0.1,63,0.1)    #in radians/sec
frequencies=frequencies[:np.size(frequency_range)] #adjust range of frequency range suitable for analysis
psd=psd_values[:,:np.size(frequency_range)] 
variance = np.zeros((N_bridge, np.size(frequency_range)))
deltaF= frequencies[4]-frequencies[3]
modal_peak_accn = np.zeros((N_bridge, np.size(t)))
for i in range(np.size(t)):
    M,K,C,_=MatrixAssemblesymetric(Human,Bridge,mped,kped,cped,xrb,length,linearMass,N_bridge,numped,t[i])
    _,accelerance = calc_frf(M,C,K,frequency_range)
    
    for j in range(1):
        for kp in range(N_bridge):
            variance[j,:] += ((np.abs(accelerance[j,kp,:]))**2)*psd[kp,:]
    variance[j, :] = variance[j, :np.size(frequencies)]
   # integrand = lambda f: np.interp(f, frequencies, variance[j, :], left=0, right=0)
   # sigma_2, _ = quad(integrand, frequencies[0], frequencies[-1])
    sigma_2=np.trapz(variance[0,:], dx=deltaF)
    sigma=np.sqrt(sigma_2)
    g_pj_value= g_pj(beamFreq,10,length,pedvelocity)
    modal_peak_accn[0,[i]]= g_pj_value*sigma


"""#
#plt.plot(t, F[2, :], label="mod 3", color='b')
#plt.plot(t, F[3, :], label="mod 4", color='orange')  # Third row

plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.title('Pedestrian Force Over Time')



fs = 1/hht 
frequencies, psd_temp = welch(F[0, :], fs=fs, nperseg=256, nfft=500,)  # Run welch once to determine size
psd_values = np.zeros((N_bridge, len(psd_temp))) 
#psd_values = np.zeros((N_bridge,np.size(t)))

# Loop through each row of matrix F (each mode) and compute its PSD
for i in range(N_bridge):  # Loop over each row (mode)
    frequencies, psd_values[[i],:] = welch(F[i, :], fs=fs, nperseg=256, noverlap=125 ,nfft=500)  # Adjust nperseg as needed
    

frequencies=frequencies[frequencies<(10)] #adjust range of frequency range suitable for analysis
psd=psd_values[:,:np.size(frequencies)]   
print(np.size(frequencies))
print(np.size(psd))

# Plot the PSD for each mode
plt.figure()  
plt.plot(frequencies, psd[0,:], label="mod 1", color='r')
#plt.plot(frequencies, psd[1,:], label="mod 2", color='g')
#plt.plot(frequencies, psd[2,:], label="mod 3", color='b')
#plt.plot(frequencies, psd[3,:], label="mod 4", color='orange')
plt.title("psd of modal forces")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.legend()
plt.grid()
plt.show()


deltaF= frequencies[4]-frequencies[3]
accnmax = np.zeros((np.size(t)))
sigma2=0

deltaF= frequencies[4]-frequencies[3]
accnmax = np.zeros((np.size(t)))
for i in range(np.size(t)-1):
    M,K,C,_=MatrixAssemblesymetric(Human,Bridge,mped,kped,cped,xrb,length,linearMass,N_bridge,numped,t[i])
    responce_std =calculate_response_std(M,C,K,frequencies,psd[0,:])
    peakfactor=g_pj(beamFreq,1,length,pedvelocity)
    accnmax[i]=peakfactor*responce_std
    xrb+=t[i+1]*pedvelocity


plt.plot(t, accnmax, label="Mode 1", color='r')  # Plot the first row
#plt.plot(t, accnmax[1, :], label="Mode 2", color='g')  # Plot the second row
plt.title("Peak Modal Acceleration")
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.grid()
plt.show()





for i in range(np.size(t)-1):
    M,K,C,_=MatrixAssemblesymetric(Human,Bridge,mped,kped,cped,xrb,length,linearMass,N_bridge,numped,t[i])
    FRF =calc_frf(M,C,K,frequencies)
    for j in range (N_bridge):
        for jk in range (N_bridge):
            sigma2+= ((FRF[[j],[jk],:]*psd[[j],:]).sum())*deltaF
        sigma =np.sqrt(sigma2) 
        peakfactor=g_pj((j+1)*beamFreq,10,length,pedvelocity)
        accnmax[[j],:]=peakfactor*sigma
        xrb+=t[i+1]*pedvelocity

plt.plot(t, accnmax[0, :], label="Mode 1", color='r')  # Plot the first row
#plt.plot(t, accnmax[1, :], label="Mode 2", color='g')  # Plot the second row
plt.title("Peak Modal Acceleration")
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.grid()
plt.show()
"""

