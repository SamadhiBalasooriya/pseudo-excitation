from matrix import*  
from solver import Newmarksuper_HSI2,accdyn_super 
from pedestrian import* 
from matplotlib import pyplot as plt
import timeit
import numpy as np
from scipy.stats import norm

# probabilitstic parameters
mean_pace = 2 #Hz  2005 pachi
pace_COV = 0.1

pace = mean_pace
DLF1 = -0.2649*pace**3 +1.3206*pace**2-1.7597*pace+0.7613

t = np.array(np.arange(0, (50+1) / 1.25, 0.01)) 

DLF2 = 0.07
DLF3 = 0.05
DLF4 = 0.05
DLF5 = 0.03 

STD_DLF2 = 0.03
STD_DLF3 = 0.02
STD_DLF4 = 0.02
STD_DLF5 = 0.015


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



def compute_dlf(f, i,ped_pace):
    
    """
    Computes the Dynamic Load Factor (DLF) for a given frequency f and index i.
    
    Parameters:
    -----------
    f : float
        The frequency for which to compute the DLF.
    i : int
        The harmonic value.
    
    Returns:
    --------
    dlf : float
        The computed DLF for the given frequency and index.
    """
    
    
    # Calculate fj = f / 2
    f_j = f /ped_pace
    
    # Get the index of the given i
    index = int(i)  # Since i is 1-based, convert it to 0-based index
    
    # Extract the corresponding values for a, b, and c
    a_i1 = a_i1_values[index]
    b_i1 = b_i1_values[index]
    c_i1 = c_i1_values[index]
    
    a_i2 = a_i2_values[index]
    b_i2 = b_i2_values[index]
    c_i2 = c_i2_values[index]
    
    a_i3 = a_i3_values[index]
    b_i3 = b_i3_values[index]
    c_i3 = c_i3_values[index]
    
    # Compute the DLF using the formula
    dlf = (a_i1 * math.exp(-((f_j - b_i1) / c_i1)**2) +
           a_i2 * math.exp(-((f_j - b_i2) / c_i2)**2) +
           a_i3 * math.exp(-((f_j - b_i3) / c_i3)**2))
    
    return dlf


def compute_subharmonic_dlf(f, i,ped_pace):
    """
    Computes the subharmonic Dynamic Load Factor (DLF) for a given frequency f and index i.
    
    Parameters:
    -----------
    f : float
        The frequency for which to compute the subharmonic DLF.
    i : int
        The index value corresponding to the table of parameters (1-based index).
    
    Returns:
    --------
    dlf : float
        The computed subharmonic DLF for the given frequency and index.
    
    Raises:
    -------
    IndexError:
        If i is outside the valid range of indices.
    """
    
    
    # Calculate fj = f / 2
    f_j = f /ped_pace
    
    
    
    # Convert i to 0-based index for accessing the lists
    index = int(i) 
    
    # Extract the corresponding values for a, b, and c
    a_s_i1 = a_s_i1_values[index]
    b_s_i1 = b_s_i1_values[index]
    c_s_i1 = c_s_i1_values[index]
    
    a_s_i2 = a_s_i2_values[index]
    b_s_i2 = b_s_i2_values[index]
    c_s_i2 = c_s_i2_values[index]
    
    # Compute the subharmonic DLF using the formula
    dlf = (a_s_i1 * math.exp(-((f_j - b_s_i1) / c_s_i1)**2) +
           a_s_i2 * math.exp(-((f_j - b_s_i2) / c_s_i2)**2))
    
    return dlf



