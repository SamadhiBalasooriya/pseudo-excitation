import math
import numpy as np
import random
from matplotlib import pyplot as plt
from pedestrian import *
from scipy.signal import welch,periodogram
from solver import Newmarkpseudo_HSI, Phi_matrix,accdyn_super
from matrix import bridge
from pseudo_excitation import *
from scipy.stats import norm

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
numbers = 3

#ped
numped = 1
pedmass = 80    #kg
peddamp = 0.3    
#pedstiff = 25000 #N/m
pedpace  = 2     #Hz
pedphase =0 
pedInlocation = 0
pedvelocity = 1.25
pedBodyF= 2 #Hz

#ped
kped=(2*np.pi*pedBodyF)**2*pedmass
cped = (2*np.pi*pedBodyF)*2*peddamp*pedmass

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
    numbers = 3,  )                   #modes


N_bridge = 3




# probabilitstic parameters
mean_pace = 2 #Hz  2005 pachi
pace_COV = 0.1

mean_alpha = np.array([0.41 * (mean_pace - 0.95),
                0.069 + 0.0056 * mean_pace,
                .033 + 0.0064 * mean_pace,
                0.013 + 0.0065 *mean_pace])

alpha_COV= np.array([0.17,0.4,0.4,0.4])


alpha_std = mean_alpha*alpha_COV


t = np.array(np.arange(0, (length+1) / pedvelocity, hht))   #for the testing length was made 10

Human = Pedestrian(
         mass = pedmass,     #kg
         damp = peddamp ,   #%
         stiff = kped, #N/m
         pace  = pedpace ,    #Hz
         phase = pedphase,
         location = pedInlocation,
         velocity = pedvelocity,
         
         iSync=0)


'''Human = Pedestrian(
         mass = randomMass,     #kg
         damp = peddamp ,   #%
         stiff = kped, #N/m
         pace  = 2 ,    #Hz
         phase = randomPhase,
         location = pedInlocation,
         velocity = randomVelocity,
         
         iSync=0)'''



# Means and standard deviation
means = [2, 4, 6, 8]
std_dev = 0.01

# Generate x values
x = np.linspace(0, 10, 100)

# Initialize a combined PDF array with zeros
combined_pdf = np.zeros_like(x)

# Plot each normal PDF
for i in range (np.size(means)):
    pdf = norm.pdf(x, means[i], std_dev)
    alpha = ((80*9.81*mean_alpha[i])**2)/2
    combined_pdf += pdf*alpha
    

frequencies = np.array(x)
psd = np.array(combined_pdf)
frequencies=frequencies[frequencies<2.5] #adjust range of frequency range suitable for analysis
psd=psd[:np.size(frequencies)]

# Plot the combined PDF
plt.plot(frequencies, psd, label='Combined PDF', color='black', linewidth=2)


# Add title and labels
plt.title('Normal PDFs with Different Means and Combined PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()

# Show the plot
plt.show()

real,imaginary = pseudo_excitation(psd,frequencies,50,1.25,t)

def worker_compute_response_pseudo(i, frequencies, t, real, Human, Bridge, numped, N_bridge, length, hht, pedvelocity, pedmass, kped, cped, linearMass, x_interested):
    """Worker function to compute pseudo excitation response for a specific frequency."""
    result = np.zeros((np.size(frequencies), np.size(t)))
    realF = np.array(real[[i], :])
    _, _, ddu = compute_response_pseudo(frequencies, t, realF, Human, Bridge, numped, N_bridge, length, hht, pedvelocity, pedmass, kped, cped, linearMass, x_interested)
    result[[i], :] = accdyn_super(Bridge, ddu, x_interested, hht)
    return result

import multiprocessing
import time

# Main code execution in stochasticPSD.py
if __name__ == "__main__":
  
    num_cores = 4 #multiprocessing.cpu_count()  # or set manually, e.g., num_cores = 4

    # Prepare the task list for parallel processing
    tasks = [( frequencies, t, real, Human, Bridge, numped, N_bridge, length, hht, pedvelocity, pedmass, kped, cped, linearMass, x_interested) for i in range(len(frequencies))]

    # Start multiprocessing
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use starmap_async to parallelize the task
        result_async = pool.starmap_async(compute_response_pseudo, tasks)

        # Optionally, do other things here while waiting for processes to finish
        print("[INFO] Parallel processing started...")
        time.sleep(1)  # Simulate doing something else

        # Wait for all processes to complete and get the results
        Real_responce = result_async.get()  # This will block until all results are available

    # Combine the results from the parallel tasks
    Real_responce1 = np.sum(Real_responce, axis=0)

    plt.plot(t, Real_responce1[0, :])
    plt.xlabel("Time")
    plt.ylabel("Response")
    plt.title("Response for first frequency")
    plt.show()

    import pickle
    with open('Real_responce_stochastic_with_HSI.pkl', 'wb') as f:
        pickle.dump(Real_responce1, f)

    print("Matrix saved to Real_responce_stochastic_with_HSI.pkl")

    print("[INFO] All processes completed.")
