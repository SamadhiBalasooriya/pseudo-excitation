import math
import numpy as np
import random
from matplotlib import pyplot as plt
from pedestrian import *
from scipy.signal import welch,periodogram
from solver import Newmarkpseudo_HSI, Phi_matrix,accdyn_super
from matrix import bridge


def pseudo_force(psd,frequency,t):
    '''uses the Power spectral dencity of the pedestrian force to create pseudo force
    psd = power spectral dencity of the input force (pedestrian force) at considered f   (N**2/Hz) the inputs psd and frequancy are arrays truncated 
    frequency =  corresponding frequancy Hz
    t= coresponding time
    returns pseudo force at a certain time'''
    psd_force = 0
    delta_f = frequency[1]-frequency[0]
    for i in range (len(frequency)):
        phi = random.uniform(0,2*np.pi)
        force=np.sqrt(2*psd[i]*delta_f)*np.cos(2*np.pi*frequency[i]*t+phi )#calculate the forces from each frequancy at a given time t.
        
        psd_force += force
        
    return psd_force

def pseudo_excitation(psd,frequency,length,velocity,t):
    '''uses the Power spectral dencity of the pedestrian force to create pseudo excitation
       psd = power spectral dencity of the input force (pedestrian force) at considered f   (N**2/Hz) the inputs psd and frequancy are arrays truncated 
       frequency =  corresponding frequancy Hz
       t= time vector
       mass = random mass of the pedestrian
       returns pseudo excitation (real and imaginary) at a certain time'''
    X_real = np.zeros((np.size(frequency),np.size(t)))
    X_imag = np.zeros((np.size(frequency),np.size(t)))
    for i in range(len(frequency)):
        X_real[[i],:]= np.sqrt(psd[i])*np.cos(2*np.pi*frequency[i]*t)
        X_imag[[i],:]= np.sqrt(psd[i])*np.sin(2*np.pi*frequency[i]*t)
    
    return X_real,X_imag



def compute_response_pseudo(frequencies,t,real,Human, Bridge,numped,N_bridge,length,hht,pedvelocity,pedmass,kped,cped,linearMass,x_interested):
    Real_responce = np.zeros((np.size(frequencies),np.size(t)))
    for i in range(np.size(frequencies)):
        realF=np.array(real[[i],:])
         #print(realF)
        _,_,ddu = Newmarkpseudo_HSI(Human, Bridge,numped,N_bridge,length,hht,pedvelocity,[pedmass],[kped],[cped],[0],linearMass,realF)
        #case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho,force
        Real_responce[[i],:]=accdyn_super(Bridge,ddu,x_interested,hht)
    return Real_responce