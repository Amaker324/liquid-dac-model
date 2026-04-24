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
from scipy.optimize import brentq

#%% THESE ARE DESIGN PARAMETERS note a lot of these are based on the numbers from the paper
L_col=7 # in meters length of column
D_col= 225 # in meters, Diamter of column


u_g=1.48 #m/s this is gas velocity. want turbulent flow but too high means its inefficeint
y_in=0.0004 #mol frac of CO2

u_l=0.005 #m/s Liquid velocity



C_OH=1100

#%%Constants
R=8.314 
MW_CO2=44.01e-3 #kg/mol
#H=29.4e5 #Henries constant Pam^3/mol CO2 in water. THIS NEEDS UPDATING
#kL=2e-4 #NOT ACCURATE
rho_l = 1000 #density of water just keeping it constant. Possible improvement here.
g = 9.81 #Gravity 

#E=500 #VERY INACCURATEW
#%% Packing Material
a=210 #packing area m^2/m^3
epsi = 0.9 #Packing void fraction
#%% Process Conditions

P_amb=101_325 #Pa

T_amb=298 #25 degrees C may change this to accept only C no Kelvin


#%% Specifications <--- Change these 

N=200 #number of discrete spots in the column
dz=L_col/N

Xs_col=np.pi*(D_col/2)**2 # Cross sectional area of column 
#Note need to add effective diameter and cross section. But thats for later
dv=dz*Xs_col

V_col=Xs_col*L_col
#%%% Stream Specifications


G_mol=(P_amb/(R*T_amb))*u_g*Xs_col



#%% Function Helpers





def calc_properties(T_K):
    """
    Calculates temperature-dependent Water Properties like density viscosity and surface tension.
    Units of mu Pa*s and sigma is N/m
    """
    
        
    #viscosity in Pa*s
    mu=1e-3*np.exp(-3.7188 + (578.919/(-137.546+T_K))) #<-- Engineers Toolbox
    #Surface tension of water (N/m) [Eq S11 form]
    tau= 1 - (T_K / 647.096)
    sigma = 0.2358 * (tau**1.256) * (1 - 0.625 * tau)
    return mu, sigma

def calc_DL(T_K, mu_Pas):
    """Calculates diffusivity of CO2 in liquid using Wilke-Chang (Eq S4)."""
   
    # Error Checked looks good
    
    Va = 34.0  #molar volume of CO2 at boiling point
    mu_cP= mu_Pas * 1000 #unit conversion
    # Result in cm^2/s from the correlation formula
    
    M_s = 18.02 #g/mol for water
    phi = 2.26 #Association parameter for water
    
    DL_cm2s = 7.4e-8 * (np.sqrt(phi * M_s) * T_K) / (mu_cP * (Va**0.6))
    return DL_cm2s / 1e4  # convert to m^2/s

def calc_aw(u_l, T_K):
    """Calculates wetted area using Onda correlation (Eq S9)."""
    # Error checked this should be good
    mu, sigma = calc_properties(T_K)
    
    # careful a here is packing specific surface area
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

def calc_kr(T=T_amb,C_i=C_OH):
    """
    Calculating rate constant. Note: T must be in range 0-50 celcius

    """
    if (0+273.15)<T<(40+273.15):
        try: 
            kr=(10**(13.635 - 2895/T))/1000
            
        except: print("Error in rate constant Calculation for 0<T<40")
    elif (20+273.15)<T<(50+273.15):
        try:
            BC=3.3968e-4*(T**2)-2.1215e-1*T+33.506
            
            k2inf=3.27869e13*np.exp(-54971/(R*T))
            
            kr=k2inf*np.exp(BC*C_i/1000)/1000
        
        except: print("Error in rate constant Calculation for 20<T<50")
    else:
        kr=1e-12
   
    return kr

def solve_E(T_K,u=u_l,C_OH=C_OH,C_i=C_OH):
    """
    Solves for E numerically.
    Note also converts C_OH to mol/L
    """
    if C_OH <=0: return 1.0
    mu,_=calc_properties(T_K)
    DL=calc_DL(T_K, mu)
    
    k2=calc_kr(T_K,C_OH)
    #print(k2)
    kL=calc_kL(T_K, u)
    M=(DL*k2*C_OH)/(kL**2)
   
    def calc_DB(T=T_K,mu=mu,C_OH=C_OH/1000):
        #this function is good. Err checked.
        
        if T<273.15: return 0
        if (1+273.15)<=T<=(25+273.15):
            K1=-7.56e-4
            K2=4.94e-6
            K3=-7.77e-9
            K4=1.1e-5
            K5=4.93e-6
            K6=-1.18e-6
            K7=-1.07
            DB=K1+K2*T+K3*(T**2)+K4*np.sqrt(T)+K5+K6*(C_OH**(3/2))+K7*np.sqrt(C_OH)/(T**2)
            return DB/(100**2)
        else:
            kb=1.38e-23 #boltzmann J/K
            d2=0.22e-9
            return kb*T/(3*np.pi*mu*d2)
        
    #end
    
    DB=calc_DB()
    
    Ei=1 + DB*C_OH/(2*DL*C_i)
    
    def E_calc(E):
        num=np.sqrt(M*(Ei-E)/(Ei-1))
        denom=max(np.tanh(num),1e-12)
        return num/denom - E
    #end
    try:
        return brentq(E_calc,1,Ei)
    except: return max(1, min(np.sqrt(M),Ei))
    






def calc_DB(T=T_amb,C_OH=C_OH):
    #this function is good. Err checked.
    mu,_=calc_properties(T)
    if T<273.15: return 0
    if (1+273.15)<=T<=(25+273.15):
        K1=-7.56e-4
        K2=4.94e-6
        K3=-7.77e-9
        K4=1.1e-5
        K5=4.93e-6
        K6=-1.18e-6
        K7=-1.07
        DB=K1+K2*T+K3*(T**2)+K4*np.sqrt(T)+K5+K6*(C_OH**(3/2))+K7*np.sqrt(C_OH)/(T**2)
        return DB/(100**2)
    else:
        kb=1.38e-23 #boltzmann J/K
        d2=0.22e-9
        return kb*T/(3*np.pi*mu*d2)

def get_Henry_Constant(T_K, C_OH,C_KCO3,C_K=C_OH):
    # 1. Pure water Henry (example correlation)
    # This represents H0 in Table S1
    H0 = (1/3.4e-2) * np.exp( (19.95e3/R)*(1/T_K - 1/298.15) ) 
    
    C_K=C_K/1000; C_OH=C_OH/1000; C_KCO3=C_KCO3/1000
    
    # 2. Salt effect (Equation 16)
    # sum(h_i * c_i) for KOH (K+ and OH-)
    # h_total = (h_K + h_G) + (h_OH + h_G)
    h_K = 0.074; h_OH = 0.066; h_G = -0.019; h_carb=0.021
    h_KOH=h_K+h_OH+h_G
    h_CO3=h_K+h_G+h_carb
    I_KOH=(1/2)*(C_K+C_OH)
    I_CO3=(1/2)*(C_K+C_KCO3)
    
    
    
    # Apply I (ionic strength) adjustment
    # log10(H/H0) = Ks * C_OH (assuming C_OH is in mol/L)
    H = H0 * 10**(h_CO3*I_CO3+h_KOH*I_KOH) # Convert your C_OH to mol/L if it's in mol/m3
    return H*101.325 # atm L/mol to Pa m^3/mol

#%% Model.
from scipy.integrate import solve_bvp

def odes(z,y):
    """
    y[0] is y_CO2 
    y[1] is C_OH concentration mol/m^3
    """
    y_CO2_val=y[0]
    C_OH_val=y[1]
    dy_dz=np.zeros_like(y)
    
    for i in range(len(z)):
        p_CO2=y_CO2_val[i]*P_amb
        C_KCO3=(C_OH-C_OH_val[i])/2 #2 of OH react to make KCO3
        H=get_Henry_Constant(T_amb, C_OH_val[i], C_KCO3)
        C_i=p_CO2/H
        
        kL=calc_kL(T_amb, u_l)
        
        E=solve_E(T_amb,C_OH=max(C_OH,1e-9), C_i=max(C_i,1e-9))
        flux=kL*C_i*E
        
        #ODEs
        dy_dz[0,i]= -(flux*a*Xs_col)/G_mol
        #liquid this one is confusing so trust????
        dy_dz[1,i]=2*flux*a/u_l
        
    return dy_dz


def boundary_cond(ya,yb):
    """
    

    Parameters
    ----------
    ya : float
        values at z=0 ie bottom of column pure air
    yb : float
        values at z=L ie top of column pure C_OH

    Returns
    -------
    RESIDUALS

    """
    res1=ya[0]-y_in
    res2=yb[1]-C_OH
    return np.array([res1,res2])




#Solver stuff

Z_vals=np.linspace(0,L_col,50)
y_guess=np.zeros((2,Z_vals.size))
y_guess[0,:]=np.linspace(y_in, y_in*0.1, Z_vals.size)
y_guess[1,:]=np.linspace(C_OH*0.8,C_OH,Z_vals.size)

sol=solve_bvp(odes, boundary_cond, Z_vals, y_guess)

if sol.success:
    z_final = np.linspace(0, L_col, 100)
    y_final = sol.sol(z_final)
    
    # 1. Grab the outlet concentrations
    y_CO2_outlet = y_final[0, -1]  # Gas mole fraction at z = L (Top)
    C_OH_outlet = y_final[1, 0]    # Liquid concentration at z = 0 (Bottom)
    
    # 2. Calculate Total Absorption (Molar)
    # Total CO2 absorbed = Gas Flow Rate * (Inlet mole fraction - Outlet mole fraction)
    total_mol_s = G_mol * (y_in - y_CO2_outlet)
    
    # 3. Convert to Mass
    total_kg_hr = total_mol_s * MW_CO2 * 3600
    total_ton_day = (total_kg_hr * 24) / 1000
    

    # 4. Removal Efficiency
    efficiency = (y_in - y_CO2_outlet) / y_in * 100

    # 5. Power
    total_power_gen=total_mol_s*(-95.8)/1000
    print("--- Absorption Results ---")
    print(f"Outlet CO2: {y_CO2_outlet:.6f} mol/mol")
    print(f"Outlet KOH: {C_OH_outlet:.2f} mol/m^3")
    print(f"Capture Rate: {total_mol_s:.4f} mol/s")
    print(f"Capture Rate: {total_kg_hr:.2f} kg/hr")
    print(f"Capture Rate: {total_ton_day:.2f} tons/day")
    print(f"Removal Efficiency: {efficiency:.2f}%")
    print(f"Heat Generation Rate: {total_power_gen:.3f} MW")
    
    plt.figure(figsize=(8, 5))
    plt.plot(z_final, y_final[0], label='Gas ($y_{CO2}$)')
    plt.ylabel('Mole Fraction')
    plt.twinx()
    plt.plot(z_final, y_final[1], color='r', label='Liquid ($C_{OH}$)')
    plt.ylabel('Concentration ($mol/m^3$)')
    plt.title('Counter-Current Column Profile (solve_bvp)')
    plt.show()
else:
    print("Solver failed:", sol.message)


#%%%OLD MODEL
# =============================================================================
# z = np.linspace(0, L_col, N+1)
# y_CO2 = np.zeros(N+1)
# C_Hydroxide=np.zeros(N+1)
# C_Hydroxide[0]=C_OH
# y_CO2[0] = y_in
# 
# capture_rate = np.zeros(N)
# 
# for i in range(N):
#     #get the partial Pressure of CO2
#     p_CO2=y_CO2[i]*P_amb
#     #Get the interfacial concentration
#     
#     H=get_Henry_Constant(T_amb, C_Hydroxide[i], C_Hydroxide[i]-C_OH)
#     
#     c_i=p_CO2/H #mol/m^3
#     
#     #Get the flux
#     kL=calc_kL(T_amb, u_l)
#     E=solve_E(T_amb,C_OH=C_Hydroxide[i],C_i=c_i)
#     
#     J=kL*c_i*E 
#     #interfacial area
#     A_i=dv*0.8*a #calc_aw(u_l, T_amb)*dv
#     #CO2 absored in the dv
#     mol_absorbed=J*A_i #mol/s
#     capture_rate[i]=mol_absorbed
#     
#     #Next gas phase
#     y_CO2[i+1]= y_CO2[i] - mol_absorbed/G_mol
#     C_Hydroxide[i+1]=C_Hydroxide[i]-2*mol_absorbed/(u_l*Xs_col*dz)
# =============================================================================

# =============================================================================
# total_capture=np.sum(capture_rate)
# removal_fraction = (y_in - y_CO2[-1]) / y_in
# 
# kg_per_hr = total_capture * 44.01e-3 * 3600
# ton_per_day=kg_per_hr/1000
# print(ton_per_day)
# print(f"Outlet CO2 mole fraction: {y_CO2[-1]:.4f}")
# print(f"Total CO2 capture rate: {total_capture:.4f} mol/s")
# print(f"Removed: {removal_fraction*100:.2f}%")
# print(f"Removal fraction: {removal_fraction:.3f}")
# 
# print(f"Amount removed in kg/hr: {kg_per_hr:.3f}")
# 
# #%% Check Results
# plt.plot(z, y_CO2)
# plt.xlabel("Column height (m)")
# plt.ylabel("CO2 mole fraction")
# plt.title("CO2 Absorption in KOH Column (Version 1)")
# plt.grid()
# plt.show()
# =============================================================================


























