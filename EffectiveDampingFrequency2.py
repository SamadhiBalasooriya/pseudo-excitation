import math
from matplotlib.ticker import ScalarFormatter
import numpy as np
import random
from matplotlib import pyplot as plt
from pedestrian import *
from scipy.signal import welch
from solver  import MatrixAssemble, MatrixAssemblesymetric  #Phi_matrix ,accdyn_super,MatrixAssemblesymetric,calculate_frf_and_accelerance,g_pj
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
pedmass = 73.85     #kg
peddamp = .3    
#pedstiff = 25000 #N/m
pedpace  = 2     #Hz
pedphase = 0
pedInlocation = 0
pedvelocity = 1.25
pedBodyF= 2.2 #Hz

#ped
kped1=14.11e3 #(2*np.pi*pedBodyF)**2*pedmass
cped1 = 612.5 #(2*np.pi*pedBodyF)*2*peddamp*pedmass

mped = np.array([pedmass/pedmass])
cped = np.array([cped1/pedmass])
kped = np.array([kped1/pedmass])


modulus =linearMass * ((2 * math.pi * beamFreq) * (math.pi / length) ** (-2)) ** 2  #E*(width*height**3)/12

#set time info
hht=0.01

#initial possition vector.......formultiple ped all these would become matrices
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

t = np.array(np.arange(0, (length + 1) / pedvelocity, hht))

Human = Pedestrian(
         mass = pedmass,     #kg
         damp = peddamp ,   #%
         stiff = kped, #N/m
         pace  = pedpace ,    #Hz
         phase = pedphase,
         location = pedInlocation,
         velocity = pedvelocity,
         
        iSync=0)


fj= np.zeros(((N_bridge+numped)*2,np.size(t)))
damp= np.zeros(((N_bridge+numped)*2,np.size(t)))


for i in range(np.size(t)):
    
    
    M, K, C,_ = MatrixAssemblesymetric(Human,Bridge,mped,kped,cped,xrb,length,linearMass,N_bridge,numped,t[i])
    # Invert the mass matrix
    M_inv = np.linalg.inv(M)

    # Number of degrees of freedom
    n_dof = M.shape[0]

    # Construct the A matrix
    A11 = np.zeros((n_dof, n_dof))  # Zero matrix for upper left
    A12 = np.eye(n_dof)             # Identity matrix for upper right
    A21 = -M_inv @ K                # -M^(-1) * K
    A22 = -M_inv @ C                # -M^(-1) * C

    # Combine into the full A matrix
    A = np.block([[A11, A12],
                    [A21, A22]])

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Compute natural frequencies (in Hz) and damping ratios
    frequencies = np.abs(eigenvalues) / (2 * np.pi)  # Natural frequencies
    fj[:,i]=frequencies.T
    damping_ratios = np.abs(eigenvalues.real) / np.abs(eigenvalues)  # Damping ratios
    damp[:,i]=damping_ratios.T
    xrb = np.add(xrb, pedvelocity*hht) #update position of pedestrian




# Plot the smoothed natural frequencies

plt.plot(t, fj[1, :])
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
plt.title("Natural Frequencies Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.grid()
plt.show()

# Plot the smoothed damping ratios

plt.plot(t, damp[1, :], label='Mode 3')
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
plt.title("Damping Ratios Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Damping Ratio")
plt.legend()
plt.grid()
plt.show()