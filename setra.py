import math
import numpy as np
import scipy
from scipy.linalg import eig
from scipy.linalg import eigh
from matrix import*
import matrix
from pedestrian import*
from sympy import *
from matplotlib import pyplot as plt
from solver import*
from matrix import*
import pedestrian

def indicat(x, lb, numped):
    """"
    Indicator function which checks whether the pedestrian is or not on the bridge, it
    returns 1 if it is on the bridge, otherwise zero.
    Parameters
    ----------
    x : vector shows the positions of wheels
        Unit m.
    lb : single span length
        Unit m.
    numped : number of pedestrians
       
        Returns
    -------
    None
    """
    I = np.zeros((numped))  # important
    for i in range(numped):
        if 0 < x[i] < lb:
            I[i] = 1
        else:
            I[i] = 0
    return I


def psi(beamFreq):
    """
    Returns the value of phi based on the structure frequency.
    
    Args:
    beamFreq (float): The structure frequency (x-axis value).
    
    Returns:
    psi: The corresponding phi value (y-axis value).
    """
    if beamFreq < 1:
        psi= 0
    elif 1 <= beamFreq < 1.7:
        # Linear interpolation from 0 to 1
        psi= (beamFreq - 1) / (1.7 - 1)
    elif 1.7 <= beamFreq <= 2.1:
        psi= 1
    elif 2.1 < beamFreq <= 2.6:
        # Linear interpolation from 1 to 0
        psi= 1 - (beamFreq - 2.1) / (2.6 - 2.1)
    else:
        psi= 0
    return psi

def setra_UDL(bridge,pedestrian,numped,t,d,beamFreq,xrb,lb):
    """
    This function is to calculate the UDL force on the bridge
    Parameters
    ----------
    pedestrian : pedestrian object
        The pedestrian object.
    bridge : bridge object
        The bridge object.
    numped: int
        Number of pedestrians.
    t : time being considered
        Unit s.
    d : pedestrian dencity 
        Unit ped/m**2.
    beamFreq : frequency of the bridge
        Unit Hz.
    Returns
    setra_UDL : at a given time t, the UDL force on the bridge
    -------
    """
    # UDL force
    # UDL force
    I = indicat(xrb, lb, numped) 
    psi_value=psi(beamFreq)
    if d==0:
        setra_udl = 280 *np.cos(2*np.pi*pedestrian.pace*t)*I   #is a moving point load/UDL
    elif d <1:
        setra_udl = d*280 *np.cos(2*np.pi*pedestrian.pace*t)*10.8 *(bridge.damp/numped)**0.5*psi_value
    else:
        setra_udl = 1*280 *np.cos(2*np.pi*pedestrian.pace*t)*1.85 *(1/numped)**0.5*psi_value

    return setra_udl

def setra_modal_force(setra_udl,length,rho):
    """
    This function is to calculate the modal force on the bridge for each considered mode Note:still this only consider one mode
    Parameters      
    ----------
    pedestrian : pedestrian object
        The pedestrian object.
    bridge : bridge object
        The bridge object.
    setra_UDL : at a given time t, the UDL force on the bridge
    numped: int
        Number of pedestrians.
    t : time being considered
        Unit s.
    d : pedestrian dencity 
        Unit ped/m**2.
    beamFreq : frequency of the bridge
        Unit Hz.
    Returns
    setra_point : at a given time t, the point force on the bridge
    -------
    """
    x=np.arange(0,length,0.1)
    #B = np.zeros((bridge.numbers,x))
    #for i in range(bridge.numbers):
    B = (2/rho/length)**0.5*np.sin(np.pi*x/length)
    #setra_udl = setra_UDL(case_bridge,case_pedestrian,numped,t,d,beamFreq)
    product = setra_udl*B

    # Apply the trapezoidal rule to integrate the product over x
    setra_modal_force= np.trapz(product, x, dx=0.1)
    
    return setra_modal_force

    
def Newmark_setra(case_bridge,case_pedestrian,N_bridge, lb, hht, v, numped,beamFreq,d,rho):
    """"
    Solve the NB matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 10) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((N_bridge, 1))
    du0 = np.zeros(( N_bridge, 1))
    ######################
    gamma = 1 / 2
    beta = 1 / 4
   
    n = np.size(t)
    h = hht

    # Constant terms and effective stiffness
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h


    Mc =bridge.Mass_matrix(self=case_bridge)
    Kc =bridge.Stiffness_matrix(self=case_bridge)
    Cc =bridge.Damp_matrix(self=case_bridge)
    Cc=np.array([Cc])

    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    #NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)
    #print(Fc)
    #bridge,pedestrian,numped,t,d,beamFreq
    xrb=[0]
    setra_udl = setra_UDL(case_bridge,case_pedestrian,numped,0,d,beamFreq,xrb,lb)
    Fc = setra_modal_force(setra_udl,lb,rho)
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((N_bridge, n))
    du = np.zeros((N_bridge, n))
    ddu = np.zeros((N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    
    for i in range(n-2):
        it = i + 1
        xrb = np.add(xrb, v * h) 
        setra_udl = setra_UDL(case_bridge,case_pedestrian,numped,t[i],d,beamFreq,xrb,lb)
        Fc = setra_modal_force(setra_udl,lb,rho)
        print(Fc)
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )
        Keff = Kc + a0 * Mc + a1 * Cc
        # u[:,[it]]=np.linalg.inv(Keff).dot(Feff)
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    
    return u, du, ddu
    
def Newmarksuper_singlesetra(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v,xrb, rho,beamFreq):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros(( N_bridge, 1))
    du0 = np.zeros((N_bridge, 1))
    ######################
    gamma = 1 / 2
    beta = 1 / 4
   
    n = np.size(t)
    h = hht

    # Constant terms and effective stiffness
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h



    Mc =bridge.Mass_matrix(self=case_bridge)
    Kc =bridge.Stiffness_matrix(self=case_bridge)
    Cc =bridge.Damp_matrix(self=case_bridge)
    Cc=np.array([Cc])
    
    
    NN=Phi_matrix(xrb,lb,rho,1,1)
    #print(Fc)
    Fc=NN*setra_UDL(case_bridge,case_pedestrian,1,t[0],0,beamFreq,xrb,lb)#np.vstack((NN*Fc, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((N_bridge, n))
    du = np.zeros((N_bridge, n))
    ddu = np.zeros((N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
        
        Mc =bridge.Mass_matrix(self=case_bridge)
        Kc =bridge.Stiffness_matrix(self=case_bridge)
        Cc =bridge.Damp_matrix(self=case_bridge)
        Cc=np.array([Cc])
        NN=Phi_matrix(xr,lb,rho,1,1)
        Fc=NN*setra_UDL(case_bridge,case_pedestrian,1,t[i],0,beamFreq,xr,lb)
       #Fc=np.vstack((NN*Fc, np.zeros((numped, 1))))
      
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )
        Keff = Kc + a0 * Mc + a1 * Cc
        # u[:,[it]]=np.linalg.inv(Keff).dot(Feff)
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    print(ddu)
    return u, du, ddu