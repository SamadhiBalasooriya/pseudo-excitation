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

#brownjohn 2004 table 1

a = [0.014, 0.119, 0.083, 0.064, 0.083, 0.115]  # A1, A2, ..., A6
b = [0.845, 0.615, 0.418, 0.431, 0.427, 0.364]  # B1, B2, ..., B6
c = [0.647, 1.213, 1.126, 1.172, 1.309, 1.674]  # C1, C2, ..., C6
d = [0.0490, 0.0034, 0.0056, 0.0054, 0.0032, 0.0007]  # D1, D2, ..., D6


t = np.array(np.arange(0, (50+1) / 1.25, 0.01))

mean_pace = 2 #Hz  2005 pachi
pace = mean_pace
f_values = [pace, 2*pace, 3*pace, 4*pace, 5*pace]
i_values = [1,2,3,4,5]
dlf_values = []
frequencies = []
x = np.arange(0, 10, 0.1)
force = np.zeros(t.size)

for i in range(np.size(i_values)):
    f = f_values[i]
    f_range = np.arange(0.95*f, 1.05*f ,0.1)
    A = a[i]
    B = b[i]
    C = c[i]
    D = d[i] 
    for j in range(np.size(f_range)):
       term = np.abs(f_range[j] / (i_values[i] * pace) - 1) ** C
       dlf = A + B * np.exp(-term / D)
       fj = f_range[j]  
       dlf_values.append(dlf)
       frequencies.append(fj)     

# Convert lists to numpy arrays for plotting
dlf_values = np.array(dlf_values)
frequencies = np.array(frequencies)

# Calculate frequency bandwidth (Δf)
#delta_f = np.mean(np.diff(frequencies))  # Mean difference between adjacent frequency values

# Calculate PSD using the equation S_j(f/f_bar) = DLF^2 / (2 * Δf)
#psd_values = (dlf_values ** 2) / (2 * delta_f)




"""
# Plot each normal PDF

psd_values = []
for i in range(np.size(i_values)):
    f = f_values[i]
    f_range = np.arange(0.95*f, 1.05*f ,1/80)
    A = a[i]
    B = b[i]
    C = c[i]
    D = d[i] 
    dlf_values_current = []

    for j in range(np.size(f_range)):
       term = np.abs(f_range[j] / (i_values[i] * pace) - 1) ** C
       dlf = A + B * np.exp(-term / D)
       fj = f_range[j]  
       dlf_values_current.append(dlf)
       frequencies.append(fj)

    
    dlf_values_current = np.array(dlf_values_current)
    dlf_values.extend(dlf_values_current)  # Append current DLF values to the main list
    
    # Calculate the normal PDF for the frequency range
    pdf = norm.pdf(f_range, f_values[i], 0.01)
    
    # Calculate the PSD using the formula: S_j = PDF * DLF^2 / 2
    psd = pdf * (dlf_values_current ** 2) / 2
    psd_values.extend(psd)  # Append PSD values to the main list

# Convert psd_values and frequencies to NumPy arrays for plotting
psd_values = np.array(psd_values)
frequencies = np.array(frequencies)"""


# Plot the result
plt.plot(frequencies, dlf_values, label=r"$G'_n(f'/f)$")
plt.title("Plot of $G'_n(f'/f)$")
plt.xlabel("Frequency $f$")
plt.ylabel(r"$G'_n(f'/f)$")
plt.grid(True)
plt.legend()
plt.show()