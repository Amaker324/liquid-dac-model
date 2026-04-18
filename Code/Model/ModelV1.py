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
kL=2e-2 #NOT ACCURATE
a=250 #packing area m^2/m^3

#%% Process Conditions

P_amb=101_325 #Pa

T_amb=298 #25 degrees C may change this to accept only C no Kelvin


#%% Specifications <--- Change these 
L_col=200 # in meters length of column
D_col= 10 # in meters, Diamter of column

N=200 #number of discrete spots in the column
dz=L_col/N

Xs_col=np.pi*(D_col/2)**2 # Cross sectional area of column 
#Note need to add effective diameter and cross section. But thats for later
dv=dz*Xs_col

#%%% Stream Specifications

u_g=2 #m/s this is gas velocity. want turbulent flow but too high means its inefficeint
y_in=0.0004 #mol frac of CO2

u_l=0.01 #m/s Liquid velocity

G_mol=(P_amb/(R*T_amb))*u_g*Xs_col

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
    J=kL*c_i #very inaaccurate changeeeee
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























