import math
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

# Regularization parameter
epsilon = 1e-4
fj= np.zeros(((N_bridge+numped),np.size(t)))
damp= np.zeros(((N_bridge+numped),np.size(t)))

for i in range(np.size(t)):
    
    
    M, K, C,_ = MatrixAssemblesymetric(Human,Bridge,mped,kped,cped,xrb,length,linearMass,N_bridge,numped,t[i])

    # Regularize M and C
    M_reg = M + epsilon * np.eye(M.shape[0])
    C_reg = C + epsilon * np.eye(C.shape[0])
    K_reg = K + epsilon * np.eye(K.shape[0])


    A11= np.zeros((np.size(M,axis=0),np.size(M,axis=1)))
    A12= np.eye(np.size(M,axis=0))
    A21= - np.linalg.pinv(M_reg) @ K_reg
    A22 = - np.linalg.pinv(M_reg) @ C_reg

    A = np.block([[A11,A12],[A21,A22]])

    #calculate eigan valeus of A
    eigenvalues, _ = np.linalg.eig(A)
    eigenvalues = eigenvalues[0::2]

    # Filter out small eigenvalues
    eigenvalues = eigenvalues[np.abs(eigenvalues) > epsilon]
    
    
    f = np.abs(eigenvalues.imag) / (2 * np.pi)
    fj[:,[i]] = f.T.reshape(-1,1)

    deltaj=abs(eigenvalues.real)
    damp[:,[i]] = deltaj.T.reshape(-1,1)/fj[:,[i]]
    xrb = np.add(xrb, pedvelocity*hht) 



""" eigenvalues, _ = np.linalg.eig(A)
    f = np.sqrt(eigenvalues.real**2 + eigenvalues.imag**2) / (2 * np.pi)
    fj[:,[i]] = f[1::2].T.reshape(-1,1)
    deltaj=abs(eigenvalues.real)
    damp[:,[i]] = deltaj[1::2].T.reshape(-1,1)/fj[:,[i]]
    
    xrb = np.add(xrb, pedvelocity* hht) """


plt.plot(t,damp[1,:])
plt.show()
plt.plot(t,fj[1,:])
plt.show()