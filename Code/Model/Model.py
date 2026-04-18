#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:01:06 2026

@author: ak
"""

'''
Everything here is subject to change. Lots of assumptions are being made etc 
etc.

'''
#%% Library time
import numpy as np
import matplotlib.pyplot as plt


#%%Constants
R=8.314 
MW_CO2=44.01e-3 #kg/mol
H=29.4e5 #Henries constant Pam^3/mol CO2 in water. THIS NEEDS UPDATING
#kL=2e-4 #NOT ACCURATE
rho_l = 1000 #density of water just keeping it constant. Possible improvement here.
g = 9.81 #Gravity 

E=500 #VERY INACCURATEW
#%% Packing Material
a=250 #packing area m^2/m^3
epsi = 0.9 #Packing void fraction
#%% Process Conditions

P_amb=101_325 #Pa

T_amb=298 #25 degrees C may change this to accept only C no Kelvin


#%% Specifications <--- Change these 
L_col=40 # in meters length of column
D_col= 10 # in meters, Diamter of column

N=200 #number of discrete spots in the column
dz=L_col/N

Xs_col=np.pi*(D_col/2)**2 # Cross sectional area of column 
#Note need to add effective diameter and cross section. But thats for later
dv=dz*Xs_col

V_col=Xs_col*L_col
#%%% Stream Specifications

u_g=2 #m/s this is gas velocity. want turbulent flow but too high means its inefficeint
y_in=0.0004 #mol frac of CO2

u_l=0.01 #m/s Liquid velocity

G_mol=(P_amb/(R*T_amb))*u_g*Xs_col






#%% Function Helpers





def calc_properties(T_K):
    """Calculates temperature-dependent Water Properties like density viscosity and surface tension."""
    
        
    #viscosity in Pa*s
    mu=1e-3*np.exp(-3.7188 + (578.919/(-137.546+T_K))) #<-- Engineers Toolbox
    #Surface tension of water (N/m) [Eq S11 form]
    tau= 1 - (T_K / 647.096)
    sigma = 0.2358 * (tau**1.256) * (1 - 0.625 * tau)
    return mu, sigma

def calc_DL(T_K, mu_Pas):
    """Calculates diffusivity of CO2 in liquid using Wilke-Chang (Eq S4)."""
    Va = 34.0  #molar volume of CO2 at boiling point
    mu_cP= mu_Pas * 1000 #unit conversion
    # Result in cm^2/s from the correlation formula
    
    M_s = 18.02 #g/mol for water
    phi = 2.26 #Association parameter for water
    
    DL_cm2s = 7.4e-8 * (np.sqrt(phi * M_s) * T_K) / (mu_cP * (Va**0.6))
    return DL_cm2s / 1e4  # convert to m^2/s

def calc_aw(u_l, T_K):
    """Calculates wetted area using Onda correlation (Eq S9)."""
    mu, sigma = calc_properties(T_K)
    
    
    Re_L = (u_l * rho_l) / (mu * a)
    Fr_L = (u_l**2 * a) / g
    We_L = (u_l**2 * rho_l) / (sigma * a)
    
    sigma_c = 0.0307 #critical surface tension N/m
    
    # Onda wetted area correlation
    ratio = 1 - np.exp(-1.45 * (sigma_c / sigma)**0.75 * (Re_L**0.1) * (Fr_L**-0.05) * (We_L**0.2))
    return a * ratio

def calc_kL(T_K, u_l):
    """
    Calculates liquid-phase mass transfer coefficient (m/s) 
    using the Onda correlation [Eq 8].
    """
    dp = 6 * (1 - epsi) / a  # effective packing diameter (m) [Eq S10]
    mu, sigma = calc_properties(T_K)
    nu = mu / rho_l     # kinematic viscosity m^2/s
    D = calc_DL(T_K, mu)   # diffusivity (m^2/s)
    aw = calc_aw(u_l, T_K) # wetted area (m^2/m^3)
    
    # Onda kL correlation
    term1 = 0.0051 * (a * dp)**0.4
    term2 = (nu * g)**(1/3)
    term3 = (u_l / (aw * nu))**(2/3)
    term4 = (nu / D)**-0.5
    
    return term1 * term2 * term3 * term4

#%% Model. Eventually make this into a function/class okay?

z = np.linspace(0, L_col, N+1)
y_CO2 = np.zeros(N+1)
y_CO2[0] = y_in

capture_rate = np.zeros(N)

for i in range(N):
    #get the partial Pressure of CO2
    p_CO2=y_CO2[i]*P_amb
    #Get the interfacial concentration
    c_i=p_CO2/H #mol/m^3
    
    #Get the flux
    kL=calc_kL(T_amb, u_l)
    J=kL*c_i*E #very inaaccurate changeeeee
    #interfacial area
    A_i=a*dv
    #CO2 absored in the dv
    mol_absorbed=J*A_i #mol/s
    capture_rate[i]=mol_absorbed
    
    #Next gas phase
    y_CO2[i+1]= y_CO2[i] - mol_absorbed/G_mol
    
total_capture=np.sum(capture_rate)
removal_fraction = (y_in - y_CO2[-1]) / y_in
    
    
kg_per_hr = total_capture * 44.01e-3 * 3600

print(f"Outlet CO2 mole fraction: {y_CO2[-1]:.4f}")
print(f"Total CO2 capture rate: {total_capture:.4f} mol/s")
print(f"Removed: {removal_fraction*100:.2f}%")
print(f"Removal fraction: {removal_fraction:.3f}")

print(f"Amount removed in kg/hr: {kg_per_hr:.3f}")

#%% Check Results
plt.plot(z, y_CO2)
plt.xlabel("Column height (m)")
plt.ylabel("CO2 mole fraction")
plt.title("CO2 Absorption in KOH Column (Version 1)")
plt.grid()
plt.show()























