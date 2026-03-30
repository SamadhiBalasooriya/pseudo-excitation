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

def Phi_x(x_interest, lb, rho, N_bridge):
    """"
    derive the mode shape of interested position at the bridge.
    Parameters
    ----------
    x_interest : interest position at the bridge.
        Unit m.
    lb : single span length
        Unit m.
    rho : unit density of bridge
        Unit kg/m.
    N_bridge : number of modes taken into account.
        
    Returns
    -------
    None
    """
    # the end
    n = N_bridge
    phi_x = np.zeros((n, 1))
    for j in range(n):
        jt = j + 1
        phi_x[j] = (1.234567901234568e-06*x_interest**4 + -0.00014814814814814815*x_interest**3 + 0.0044444444444444444*x_interest**2)
    return phi_x




def Phi_matrix(xrb, lb, rho, N_bridge,numped):
    """"
    derive the mode shape of interested position at the bridge.
    Parameters
    ----------
    x : position of pedestrians with time
        Unit m.
    lb : single span length
        Unit m.
    rho : unit density of bridge
        Unit kg/m.
    N_bridge : number of modes taken into account.
        
    Returns
    -------
    None
    """
   
    n = N_bridge
    N = np.zeros((n, numped))
    for i in range(n):
        for j in range(numped):
          it = i + 1
          N[i][j] = (1.234567901234568e-06*(xrb[j])**4 + -0.00014814814814814815*(xrb[j])**3 + 0.0044444444444444444*(xrb[j])**2) #x interest should be time dependednt x=vt xrb is a matrix with time varying location of each pedestrian
    I = indicat(xrb, lb, numped) 
    NN = np.dot(N, np.diag(I))
    return NN
    

def MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,t):
    """assembles the matrices with coupled SMD properties. Coupling is in M here"""
    #mped kped cped are the individual property containing matrices

    mass=bridge.Mass_matrix(self=case_bridge)
    k= bridge.Stiffness_matrix(self=case_bridge)
    c=bridge.Damp_matrix(self=case_bridge)
    NN = Phi_matrix(xrb, lb, rho, N_bridge,numped)
    v=Pedestrian.detVelocity
    
    #M assemble
    M3 =np.diag(mped)
    M1= np.hstack((mass,NN*M3)) 
    M2= np.zeros((numped,N_bridge))#(diag(mped).shape)
    M4 = np.hstack((M2,M3))
    M = np.vstack((M1,M4))
    #end

    #K assemble
    K2= np.zeros((N_bridge,numped))#(diag(k).shape)
    K3= np.hstack((k,K2))
    K4 = np.diag(kped)
    K5=np.hstack((-np.array(NN*K4).T,K4))
    K=np.vstack((K3,K5))
    #end

    #C assemble
    C2= np.zeros((N_bridge,numped))
    C3= np.hstack((c,C2))
    C4 = np.diag(cped)
    C5=np.hstack((-np.array(NN*C4).T,C4))
    C=np.vstack((C3,C5))
    #end
    
    #F matrix
    Ft = pedestrian.calcPedForce(case_pedestrian, t)  #this should get a set of arrays and choose the value for corresponding xrb
 
    
    #end
    return M,K,C,Ft


def Newmarksuper_HSI(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
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



    Mc, Kc, Cc, Fc = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)

    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)
    #print(Fc)
    Fc=np.vstack((NN*Fc, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
        
        Mc, Kc, Cc, Fc = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
        NN=Phi_matrix(xr,lb,rho,N_bridge,numped)
       
        Fc=np.vstack((NN*Fc, np.zeros((numped, 1))))
      
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
    

def accdyn_super(bridge_instance,ddu, x_inter, hht):
    """
    Generate the acceleration vector (time) of the interested point
    at the bridge.
    Parameters
    ----------
    ddu : acceleration of the HSI system.
        Unit m/s^2.
    
    x_inter : dynamic of interest point at the bridge.
        Unit m.
    N_span : number of spans in the bridge.
        1 means single span
    v : pedestrian speed
        m/s.
    hht : time steps
        Unit s.
    -------
    None.
    """
    lb = bridge_instance.L
    rho = bridge_instance.rho
    N_bridge = bridge_instance.n
    v=Pedestrian.detVelocity

    phi_x = Phi_x(x_inter, lb, rho, N_bridge)
   
   
    meta = (np.diag(phi_x.flatten()).dot(ddu[:N_bridge, :]))
    

    column_sums = np.sum(meta, axis=0) 

    return column_sums





'''for pseudo excitation method'''

def Newmarkpseudo_HSI(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho,force):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
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


    Mc, Kc, Cc, _ = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)
    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    
    NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)

    
    Fc=np.vstack((NN*0, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
       
        Mc, Kc, Cc,_= MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
        NN=Phi_matrix(xr,lb,rho,N_bridge,numped)
        #print(Fc[it])
        Fc=np.vstack((NN*force[:,[i]], np.zeros((numped, 1))))
       
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

def Newmarkpseudo_HSI2(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho,force):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
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


    Mc, Kc, Cc, _ = MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)
    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    
    NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)

    
    Fc=np.vstack((NN*0, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
       
        Mc, Kc, Cc,_= MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
        NN=Phi_matrix(xr,lb,rho,N_bridge,numped)
        #print(Fc[it])
        Fc=np.vstack((NN*force[:,[i]], np.zeros((numped, 1))))
       
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

def MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,t):
    """assembles the matrices with coupled SMD properties. Coupling is in K and C here"""
    #mped kped cped are the individual property containing matrices

    mass=bridge.Mass_matrix(self=case_bridge)
    k= bridge.Stiffness_matrix(self=case_bridge)
    c=bridge.Damp_matrix(self=case_bridge)
    NN = Phi_matrix(xrb, lb, rho, N_bridge,numped)
    v=Pedestrian.detVelocity
    kped = np.array(kped)
    cped = np.array(cped)
    #mass matrix
    M1 =np.diag(mped)
    M2= np.zeros((N_bridge,numped))
    M3= np.hstack((mass,M2)) 
    M4= np.hstack((M2.T,M1))
    M = np.vstack((M3,M4))
    #end

    #stiffness matrix
    
    K1 = np.diag(kped)
    K2 = -kped*NN
    K3 = -(NN.T)*kped[:,np.newaxis]   #-K1*(NN.T)
    K4= np.hstack((K3,K1))
    for i in range(numped):
        k += kped[i]*(NN[:,[i]])*np.tile(NN[:,[i]].reshape(1, -1), (N_bridge, 1))
    K5 = np.hstack((k,K2))
    K = np.vstack((K5,K4))
    #end

    #C assemble
    C1 = np.diag(cped)
    C2 = -cped*NN
    C3 = -(NN.T)*cped[:,np.newaxis]
    C4= np.hstack((C3,C1))
    for i in range(numped):
        c += cped[i]*(NN[:,[i]])*np.tile(NN[:,[i]].reshape(1, -1), (N_bridge, 1))
    C5 = np.hstack((c,C2))
    C = np.vstack((C5,C4))
    #end

    Ft = pedestrian.calcPedForce(case_pedestrian, t)  #this should get a set of arrays and choose the value for corresponding xrb
    Ft=Ft*NN
    Fsum= np.sum(Ft,axis=1)
    F = Fsum[:, np.newaxis]
  
    return M,K,C,F
  
def Newmarksuper_HSI2(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
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

   

    Mc, Kc, Cc, Fc = MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)
   
    Fc=np.vstack((Fc, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
    
        Mc, Kc, Cc, Fc = MatrixAssemblesymetric(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
      
        Fc=np.vstack((Fc, np.zeros((numped, 1))))
        
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


def calc_frf(M, C, K, freq_range,):
    """
    Calculate the |Frequency Response Function (FRF)|**2 for a MDOF system.
    
    Parameters:
    M :Mass matrix.

    C :Damping matrix.

    K :Stiffness matrix.

    freq_range : ndarray Array of frequencies at which to calculate the FRF.(rad/s note:that not in Hz)
    
    t= timepoint being considerd 
    
    Returns:
    FRF : ndarray
        Frequency Response Function matrix, each column corresponds to a frequency.
    """
    n_dof = M.shape[0]
    FRF = np.zeros((n_dof, n_dof, len(freq_range)), dtype=complex)
    accelerance = np.zeros((n_dof, n_dof, len(freq_range)), dtype=complex)
    
    for i, omega in enumerate(freq_range):
        omega_squared = omega**2
        H = K - omega_squared * M + 1j * omega * C  # Complex dynamic stiffness matrix
        FRF[:, :, i] = np.linalg.inv(H)
        accelerance [:, :, i] = -omega_squared * FRF[:, :, i]

    return FRF, accelerance
    
    


def g_pj(omega_j, N, L, vm):

    """
    Calculate the g_pj value based on the given parameters.

    This function computes the g_pj value, which is typically used in the analysis of
    structural responses. The g_pj value is calculated using the frequency `fe`, 
    the period `T`, and the given angular frequency `ωj`.

    Parameters:
    -----------
    omega_j : float
        Angular frequency (ωj) in radians per second.
    N : int or float
        Number of cycles.
    L : float
        Length in meters.
    vm : float
        Velocity in meters per second.

    Returns:
    --------
    g_pj_value : float
        The computed g_pj value based on the provided input parameters.
    """

    # Calculate fe
    fe = omega_j / (2 * np.pi)
    
    # Calculate T
    T = (N * L) / vm
    
    # Calculate g_pj
    g_pj_value = (np.sqrt(2 * np.log(2* fe * T)) + 0.5772 / np.sqrt(2 * np.log(2 * fe * T))) 
    
    return g_pj_value

def calculate_frf_and_accelerance(M, C, K, frequencies):
    """
    Calculate the Frequency Response Function (FRF) and accelerance for given M, C, K matrices over a range of frequencies.

    Parameters:
    M (numpy.ndarray): Mass matrix
    C (numpy.ndarray): Damping matrix
    K (numpy.ndarray): Stiffness matrix
    frequencies (numpy.ndarray): Array of frequencies

    Returns:
    tuple: (FRF, accelerance) where both are numpy.ndarrays of shape (len(frequencies), M.shape[0], M.shape[1])
    """
    # Initialize arrays to store the FRF and accelerance values
    FRF = np.zeros((len(frequencies), M.shape[0], M.shape[1]), dtype=complex)
    accelerance = np.zeros((len(frequencies), M.shape[0], M.shape[1]), dtype=complex)

    # Loop over each frequency to calculate the FRF and accelerance
    for i, freq in enumerate(frequencies):
        omega = 2 * np.pi * freq  # Angular frequency
        H = np.linalg.inv(K - omega**2 * M + 1j * omega * C)  # Calculate the FRF
        FRF[i, :, :] = H
        accelerance[i, :, :] = -omega**2 * H

    return FRF, accelerance

def calculate_response_std(M, C, K, frequencies, input_psd,n_dof):
    """
    Calculate the response_std given the input PSD and the M, C, K matrices over a range of frequencies.

    Parameters:
    M (numpy.ndarray): Mass matrix
    C (numpy.ndarray): Damping matrix
    K (numpy.ndarray): Stiffness matrix
    frequencies (numpy.ndarray): Array of frequencies
    input_psd (numpy.ndarray): Input PSD array of shape (len(frequencies),)

    Returns:
    numpy.ndarray: Response PSD array of shape (len(frequencies), M.shape[0], M.shape[1])
    """
    # Calculate the FRF and accelerance
    _, accelerance = calc_frf(M, C, K, frequencies)

    # Calculate the magnitude squared of the FRF
    accelerance_magnitude_squared = np.abs(accelerance)**2
    #FRF_magnitude_squared = np.abs(FRF)**2

    # Calculate the response_std
    deltaF= frequencies[4]-frequencies[3]
    #n_dof = M.shape[0]
    for i in range(n_dof):
        multiply = accelerance_magnitude_squared[[i],[i],:].flatten() * input_psd[[i],:]
    
    sigma2=np.trapz(multiply, dx=deltaF, axis=0)
    
    
    return np.sqrt(sigma2)

def plot_frf_magnitude(M, C, K, freq_range):
    """
    Plot the FRF magnitude of a multi-degree-of-freedom system.
    
    Parameters:
    M : ndarray
        Mass matrix.
    C : ndarray
        Damping matrix.
    K : ndarray
        Stiffness matrix.
    freq_range : ndarray
        Array of frequencies at which to calculate the FRF.
    
    Returns:
    None. The function plots the FRF magnitude for each degree of freedom.
    """
    n_dof = M.shape[0]
    FRF = np.zeros((n_dof, n_dof, len(freq_range)), dtype=complex)
    
    for i, omega in enumerate(freq_range):
        omega_squared = omega**2
        H = K - omega_squared * M + 1j * omega * C  # Complex dynamic stiffness matrix
        FRF[:, :, i] = np.linalg.inv(H)
    
    # Plot the magnitude of the FRF for each degree of freedom
    #plt.figure(figsize=(10, 6))
    for dof in range(n_dof):
        plt.plot(freq_range, np.abs(FRF[dof, dof, :]), label=f'DOF {dof + 1}')
    
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('FRF Magnitude')
    plt.title('Frequency Response Function (FRF) Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()


def montecarlo_stocastic_accn(length, modulus, linearMass, modalDampingRatio, numbers, pedmass, peddamp, pedBodyF, pedvelocity, numped, hht, x_interested,i):
    import random
    acceleration_responses = np.zeros((100, len(x_interested)))
    mean_pace = 2 #Hz  2005 pachi
    pace_COV = 0.01

    mean_mass= 70 #kg
    mass_COV= 0.17 #from butz 2008

    mean_velocity = 1.3
    std_velocity = 0.12 #pachi 2005

    # Randomize pedestrian parameters: mass, and pace
    randomPace = random.gauss(mean_pace, pace_COV*mean_pace)
    randomMass = random.gauss(mean_mass, mass_COV*mean_mass) # Damping ratio (mean=0.3, std=0.05)
    
    # Calculate pedestrian stiffness and damping based on random parameters
    kped = (2 * np.pi * pedBodyF) ** 2 * mean_mass
    cped = (2 * np.pi * pedBodyF) * 2 * peddamp * pedmass

    # Convert to arrays for compatibility with the solver
    mped = np.array([pedmass])
    cped = np.array([cped])
    kped = np.array([kped])
    xrb = [0]  # Initial position

    # Create bridge and pedestrian instances
    Bridge = bridge(
            length=length,
            modulus=modulus,
            density=linearMass,
            damp=modalDampingRatio,
            numbers=numbers
                        )

    Human = Pedestrian(
            mass=pedmass,
            damp=peddamp,
            stiff=kped,
            pace=randomPace,
            phase=0,
            location=0,
            velocity=pedvelocity,
            iSync=0
        )

    # Solve for acceleration response with Human-Structure Interaction (HSI)
    _, _, ddu_hsi = Newmarksuper_HSI(Human, Bridge, numped, numbers, length, hht, pedvelocity, mped, kped, cped, xrb, linearMass)
    accn_hsi = accdyn_super(Bridge, ddu_hsi, x_interested, hht)

    # Store the acceleration response for this simulation
    acceleration_responses[i, :] = accn_hsi

    return acceleration_responses