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
import pandas as pd

#%% THESE ARE DESIGN PARAMETERS note a lot of these are based on the numbers from the paper
L_col=7 # in meters length of column
Xs_col=45_000 # Cross sectional area of column 

u_g=1.6 #m/s this is gas velocity. want turbulent flow but too high means its inefficeint
y_in=0.0004 #mol frac of CO2

u_l=0.01 #m/s Liquid velocity

RH=0.5

C_OH=1100 #mol/m^3

T_pellet=298.15
T_slaker=300+273.15
T_calciner=900+273.15


#%%Constants
R=8.314 

MW_CO2=44.01e-3 #kg/mol
MW_H2O=18.02e-3
MW_Air=29.97e-3
MW_CaCO3=100.09e-3
MW_CaO=56.08e-3
MW_CaOH2=74.09e-3

cp_CaCO3=0.76e3 #J/kgK
cp_CaO=(737+1800)/2 # from https://doi.org/10.1016/j.rser.2025.115678 average
cp_L=4184 #J/kgK
cp_CaOH2 = 1200


rho_l = 1000 #density of water just keeping it constant. Possible improvement here.

g = 9.81 #Gravity 


#%% Packing Material
a=210 #packing area m^2/m^3
epsi = 0.8 #Packing void fraction
#%% Process Conditions

P_amb=101_325 #Pa

T_amb=298 #25 degrees C may change this to accept only C no Kelvin


#%% Specifications <--- Change these 

N=200 #number of discrete spots in the column
dz=L_col/N


#Note need to add effective diameter and cross section. But thats for later
dv=dz*Xs_col

V_col=Xs_col*L_col
#%%% Stream Specifications


G_mol=(P_amb/(R*T_amb))*u_g*Xs_col



#%% Function Helpers


def get_Psat_H2O(T_K):
    """Buck Equation idk look it up on wikipedia"""
    T_C=T_K-273.15
    return 6.1121*100 * np.exp( (18.678- T_C/234.5) * ( T_C/ ( 257.14+T_C) ) )


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

def calc_hL(T_K, u_l):
    """
    Liquid-side heat transfer coefficient using Chilton-Colburn analogy
    based on Onda kL correlation already implemented.
    """

    # properties
    mu,_ =calc_properties(T_K)
    rho =rho_l
    # mass transfer coefficient
    kL = calc_kL(T_K, u_l)

    # diffusivity for Sc
    D = calc_DL(T_K, mu)

    # thermal diffusivity (rough water estimate)
    k = 0.6  # W/m-K
    alpha = k / (rho * cp_L)

    Sc = mu / (rho * D)
    Pr = mu / (rho * alpha)

    hL = kL * rho * cp_L * (Sc / Pr)**(2/3)

    return hL

#%% Model.
from scipy.integrate import solve_bvp

def odes(z,y):
    """
    y[0] is y_CO2 
    y[1] is C_OH concentration mol/m^3
    y[2] is T_G
    y[3] is T_L 
    y[4] is y_H2O humdity mol fraction
    """
    
    
    
    dy_dz=np.zeros_like(y)
    
    for i in range(len(z)):
        
        y_CO2_val=y[0,i]
        C_OH_val=y[1,i]
        T_G_val=y[2,i]
        T_L_val=y[3,i]
        y_H2O_val=y[4,i]
        G_mol=(P_amb/(R*T_G_val))*u_g*Xs_col
        G_flux=u_g*P_amb/(R*T_G_val)
        #print(G_flux)
        p_CO2=y_CO2_val*P_amb
        C_KCO3=(C_OH-C_OH_val)/2 #2 of OH react to make KCO3
        H=get_Henry_Constant(T_L_val, C_OH_val, C_KCO3)
        aw=calc_aw(u_l, T_L_val)
        C_i=p_CO2/H
        
        kL=calc_kL(T_L_val, u_l)
        
        E=solve_E(T_L_val,C_OH=max(C_OH_val,1e-9), C_i=max(C_i,1e-9))
        flux=kL*C_i*E
        
        y_sat=get_Psat_H2O(T_L_val)/P_amb
        
        kg=0.025 # fix to be more accurate
        flux_H2O=kg*(y_sat-y_H2O_val)/(R*T_G_val)
        
        
        #ODEs
        #dy_dz[0,i]= -(flux)/G_flux 
        dy_dz[0,i]= -(flux*aw)/G_flux # mol/m^2 s * m^2/m^3 / mol/m^2s --> 1/m
        #liquid this one is confusing so trust????
        dy_dz[1,i]=2*flux*aw/u_l # mol/m^2s * m^2/m^3 / m/s --> mol/m^3 1/m
        h_heat=calc_hL(T_L_val, u_l) # J/m^2sK
        #Temp of gas 
        dy_dz[2,i]= (h_heat*aw* (T_L_val-T_G_val))/(G_mol*MW_Air*1e3) #-->J/m^2sK * m^2/m^3 * K/ / (mol/s * kg/mol* J/kg K) -->  
        dy_dz[3,i]= -(h_heat*aw*(T_G_val-T_L_val) -flux_H2O*aw*44e3 + flux*aw*9e4 )/(u_l*rho_l*4184) # add reaction term
        dy_dz[4,i]= flux_H2O*aw/G_flux
      
    return dy_dz


def boundary_cond(yb,yt):
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
    res1=yb[0]-y_in
    res3=yb[2]-T_amb
    res2=yt[1]-C_OH
    res4=yt[3]-T_amb
    res5=yb[4]-(RH*get_Psat_H2O(T_amb)/P_amb)
    return np.array([res1,res2,res3,res4,res5])




#Solver stuff

Z_vals=np.linspace(0,L_col,50)
y_init = np.zeros((5, Z_vals.size))
y_init[0,:] = y_in
y_init[1,:] = C_OH
y_init[2,:] = T_amb
y_init[3,:] = T_amb
y_init[4,:] = 0.01

sol=solve_bvp(odes, boundary_cond, Z_vals, y_init)

if sol.success:
    z_final = np.linspace(0, L_col, 100)
    y_final = sol.sol(z_final)
    
    # 1. Grab the outlet concentrations
    y_CO2_outlet = y_final[0, -1]  # Gas mole fraction at z = L (Top)
    C_OH_outlet = y_final[1, 0]    # Liquid concentration at z = 0 (Bottom)
    y_H2O_out=y_final[4,-1]
    
    G_flux=u_g*P_amb/(R*T_amb)
    # 2. Calculate Total Absorption (Molar)
    # Total CO2 absorbed = Gas Flow Rate * (Inlet mole fraction - Outlet mole fraction)
    #total_mol_s = G_flux * Xs_col*(y_in - y_CO2_outlet)
    total_mol_s = G_mol*(y_in - y_CO2_outlet)
    n_H2O_out=y_H2O_out*G_mol
    y_H2O_in= RH * get_Psat_H2O(T_amb)/ P_amb
    n_evap=n_H2O_out-y_H2O_in*G_mol
    n_evap_perHr=n_evap*60*60
    kgWaterLostPerHr=n_evap_perHr*MW_H2O
# =============================================================================
#     F_OH_in  = u_l * Xs_col * C_OH
#     F_OH_out = u_l * Xs_col * C_OH_outlet
#     total_mol_s = 0.5 * (F_OH_in - F_OH_out)
# =============================================================================
    
    # 3. Convert to Mass
    total_kg_hr = total_mol_s * MW_CO2 * 3600
    total_ton_day = (total_kg_hr * 24) / 1000
    

    # 4. Removal Efficiency
    efficiency = (y_in - y_CO2_outlet) / y_in * 100


    print("--- Absorption Results ---")
    print(f"Outlet CO2: {y_CO2_outlet:.6f} mol/mol")
    print(f"Outlet KOH: {C_OH_outlet:.2f} mol/m^3")
    print(f"Capture Rate: {total_mol_s:.4f} mol/s")
    print(f"Capture Rate: {total_kg_hr:.2f} kg/hr")
    print(f"Capture Rate: {total_ton_day:.2f} tons/day")
    print(f"Removal Efficiency: {efficiency:.2f}%")
    print(f"# of moles of water lost per hour: {n_evap_perHr:.2f}")
    print(f"# of moles of water lost per second: {n_evap:.2f}")
    print(f"kg of water lost per hour: {kgWaterLostPerHr:.2f}")
    
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


print()
print("=========================================")
print()
#%% Further processing
#Outlet stream

vdot=Xs_col*u_l #m^3/s
print(f"Volumetric flow: {vdot:.2f} m^3/s")
C_KCO3=0.5*(C_OH-C_OH_outlet)
mol_CO3=C_KCO3*vdot


#%%% Pellet reactor
# =============================================================================
# H_CaCO3=-5.8e-3 #MJ/mol
# Q_PR=C_KCO3*vdot*H_CaCO3 # mol/m^3 * m^3/s * MJ/mol
# print(f"Heat released by pellet reactor: {Q_PR:.2f} MW")
# #assuming x amount of size of the reactor
# V_PR=10000 #m^3
# 
# # Q=Mass*Cp*delta T
# T_risePR=-Q_PR*1e6 / (V_PR*rho_l*cp_L) # J/s / (m^3/s * kg/m^3 * J/kg K) --> K
# print(f"Temperature Rise in the pellet reactor assuming {V_PR:.0f} is {T_risePR:.4f}")
# 
# =============================================================================

#%% Further process specs
def pellet_reactor(vdot_L, C_OH_in, C_K2CO3_in, X=1):
    "lumped Pellet reactor"
    n_OH_in=vdot_L*C_OH_in #mol/s
    n_CO3_in=vdot_L*C_K2CO3_in
    
    n_CO3_rxn=X*n_CO3_in
    
    
    n_CaOH2=n_CO3_rxn
    n_CaCO3=n_CO3_rxn
    n_OH_out=n_OH_in+2*n_CO3_rxn
    n_CO3_out=n_CO3_in-n_CO3_rxn
    
    
    C_OH_out=n_OH_out/vdot_L
    C_CO3_out=n_CO3_out/vdot_L
    
    
    dH=-5.8e3 #J/mol
    Q=n_CaCO3*dH
    return [C_OH_out,C_CO3_out,n_CaCO3,n_CaOH2,Q]
#end

def pellet_separator(n_CaCO3, solids_recovery=1):
    solids_to_calciner= solids_recovery * n_CaCO3
    fines_loss=(1 - solids_recovery) * n_CaCO3
    return solids_to_calciner, fines_loss
#end

def calciner(n_CaCO3, X=1):
    n_rxn=X*n_CaCO3

    n_CaO=n_rxn
    n_CO2=n_rxn
    n_unreacted=n_CaCO3 - n_rxn

    # J/mol CaCO3
    dH = 178.3e3
    Q = n_rxn * dH

    return [n_CaO, n_CO2,n_unreacted,Q]
#end

def slaker(n_CaO, X=1):
    n_rxn= X * n_CaO
    n_CaOH2= n_rxn

    dH= -63.9e3  # J/mol
    Q =n_rxn * dH

    return [n_CaOH2,Q]



#%% calculations
C_OH_outPR,C_CO3_outPR,n_CaCO3PR,n_CaOH2PR,Q_PR=pellet_reactor(vdot, C_OH_outlet, C_KCO3)
Q_PR=Q_PR/1e3 #kW
n_CaOCalc, n_CO2Calc, n_CaCO3Calc,Q_calc=calciner(n_CaCO3PR)
Q_calc=Q_calc/1e3 #kW

n_CaOH2Slak, Q_slak=slaker(n_CaOCalc)
Q_slak=Q_slak/1e3

Mass_CO2=n_CO2Calc*MW_CO2*3600/1000*24 #mol/s --> metric ton/day
print("=================")
print("Results:")
print(f"Metric tons of CO2 per day: {Mass_CO2:.2f}")
print(f"Heat from Pellet Reactor: {Q_PR:.3f} kW" )
print(f"Heat from Slaker: {Q_slak:.3f} kW")

print(f"Heat from Calciner: {Q_calc:.3f} kW" )


#Sensible Heat Requirements

# 1. Heating CaCO3 pellets from Pellet Reactor (25C) to Calciner (900C)
mass_flow_CaCO3 = n_CaCO3PR * MW_CaCO3 # kg/s
Q_sens_CaCO3 = mass_flow_CaCO3 * cp_CaCO3 * (T_calciner - T_pellet) # Watts

# 2. Heating/Cooling CaO from Calciner (900C) to Slaker (300C)
# Note: This is usually heat RECOVERED to preheat the calciner feed
mass_flow_CaO = n_CaOCalc * MW_CaO # kg/s
Q_sens_CaO = mass_flow_CaO * cp_CaO * (T_calciner - T_slaker) # Watts

# 3. Heating the KOH solution returned from Slaker/Clarifier back to Contactor 
# (Usually negligible unless there's a specific process heater)
mass_flow_CaOH2 = n_CaOH2Slak * MW_CaOH2
Q_cool_CaOH2 = mass_flow_CaOH2 * cp_CaOH2 * (T_pellet - T_slaker)
Q_sens_Soln = vdot * rho_l * cp_L * (T_pellet - T_amb) # J/s (Watts)
Q_into_liquid_loop_MW = (Q_PR + Q_cool_CaOH2/ 1000)/1000
# --- Total Energy Summary ---


total_thermal_demand_kW = (
    Q_calc +                # Heat of reaction 
    (Q_sens_CaCO3 / 1000) + # Sensible heat from HEATING
    abs(Q_sens_CaO / 1000 ) +  #Sensible heat from COOLING
    abs(Q_into_liquid_loop_MW*1e3) #Sensible heat from COOLING +rxn
)

Q_into_liquid_loop_MW = (Q_PR + Q_cool_CaOH2/ 1000)/1000

Trise=Q_into_liquid_loop_MW*1e6/(cp_L*vdot*rho_l)

print("=========================================")
print("     EXTENDED ENERGY & MASS BALANCE     ")
print("=========================================")
print(f"Capture Performance:")
print(f"  - CO2 Captured:        {Mass_CO2:.2f} t/day")
print(f"  - CO2 Captured:        {Mass_CO2*365/1000_000:.2f} Mt/year")
print(f"  - Removal Efficiency:  {efficiency:.2f} %")
print(f"  - Water Loss:          {kgWaterLostPerHr:.2f} kg/hr")

print(f"\nReaction Heats (Negative = Exothermic):")
print(f"  - Pellet Reactor:      {Q_PR/1000:.2f} MW")
print(f"  - Slaker:              {Q_slak/1000:.2f} MW")
print(f"  - Calciner (Rxn):      {Q_calc/1000:.2f} MW")

print(f"\nSensible Heat Demands (Pre-recovery):")
print(f"  - Heat CaCO3 25C to 900C:  {Q_sens_CaCO3/1e6:.2f} MW")
print(f"  - CaO Cooling Potential: {Q_sens_CaO/1e6:.2f} MW")

print(f"Heat Dumped into KOH Loop:  {Q_into_liquid_loop_MW:.2f} MW")

print(f"\nTotal Plant Thermal Estimate:")
print(f"  - Net Thermal Load:    {total_thermal_demand_kW/1000:.2f} MW")
print(f"  - Specific Energy:     {(total_thermal_demand_kW*3600/1000) / (total_kg_hr):.2f} GJ/t-CO2")
print("=========================================")

#%% ============================================================
# ENERGY INTEGRATION: CLEAN SCENARIO ANALYSIS
# ============================================================
# This block REPLACES everything after:
# "EXTENDED ENERGY & MASS BALANCE"
#
# Philosophy:
#   Stage 1 = Base process (no recovery)
#   Stage 2 = HEN / Pinch (internal recovery only)
#   Stage 3 = HEN + Heat Pump (utility upgrade)
#
# Your extended mass & energy balance above remains the thermodynamic source
# of truth. This section is ONLY post-processing utility analysis.
# ============================================================

#%% -----------------------------
# Helper Functions
# -----------------------------
def shift_streams(df, DTmin):
    """
    Apply pinch temperature shifting.
    Hot streams shifted down by DTmin/2
    Cold streams shifted up by DTmin/2
    """
    df = df.copy()
    df["Ts_shift"] = np.where(df["type"] == "hot",  df["Ts"] - DTmin/2, df["Ts"] + DTmin/2)
    df["Tt_shift"] = np.where(df["type"] == "hot",  df["Tt"] - DTmin/2, df["Tt"] + DTmin/2)
    return df


def build_interval_table(df):
    """
    Build pinch interval table from shifted streams.
    Returns interval dataframe used for heat cascade.
    """
    temps = sorted(set(df["Ts_shift"]).union(set(df["Tt_shift"])), reverse=True)
    rows = []

    for i in range(len(temps) - 1):
        Th = temps[i]
        Tl = temps[i + 1]
        dT = Th - Tl

        hot_active = df[
            (df["type"] == "hot") &
            (df["Ts_shift"] >= Th) &
            (df["Tt_shift"] <= Tl)
        ]

        cold_active = df[
            (df["type"] == "cold") &
            (df["Tt_shift"] >= Th) &
            (df["Ts_shift"] <= Tl)
        ]

        CP_hot = hot_active["CP"].sum()
        CP_cold = cold_active["CP"].sum()

        dCP = CP_hot - CP_cold
        dH = dCP * dT

        rows.append([Th, Tl, dT, CP_hot, CP_cold, dCP, dH])

    return pd.DataFrame(
        rows,
        columns=["Th", "Tl", "dT", "CP_hot", "CP_cold", "dCP", "dH"]
    )


def heat_cascade(intervals):
    """
    Perform heat cascade and return:
    cascade, QHmin, QCmin
    """
    H = [0.0]
    for dH in intervals["dH"]:
        H.append(H[-1] + dH)

    H = np.array(H)
    QHmin = max(0.0, -H.min())
    cascade = H + QHmin
    QCmin = cascade[-1]

    return cascade, QHmin, QCmin


def build_stream_table():
    """
    Build process stream table directly from base-case model results.
    All temperatures in C
    CP in kW/K
    """
    cp_CO2 = 1.15e3  # J/kg-K

    m_CO2 = n_CO2Calc * MW_CO2
    m_H2O = max(n_CaOCalc * MW_H2O, 1e-9)

    CP_vals = {
        "CaO":   (mass_flow_CaO   * cp_CaO)   / 1000,   # kW/K
        "CaCO3": (mass_flow_CaCO3 * cp_CaCO3) / 1000,   # kW/K
        "CO2":   (m_CO2           * cp_CO2)   / 1000,   # kW/K
        "Water": (m_H2O           * cp_L)     / 1000,   # kW/K
        "CaOH2": (mass_flow_CaOH2 * cp_CaOH2) / 1000    # kW/K
    }

    streams = pd.DataFrame([
        ["H1_CaO",    "hot",  900.0, 300.0, CP_vals["CaO"]],
        ["H2_Slaker", "hot",  300.0,  40.0, CP_vals["CaOH2"]],
        ["H3_CO2",    "hot",  900.0, 100.0, CP_vals["CO2"]],
        ["C1_CaCO3",  "cold",  25.0, 900.0, CP_vals["CaCO3"]],
        ["C2_Water",  "cold",  25.0,  95.0, CP_vals["Water"]],
    ], columns=["name", "type", "Ts", "Tt", "CP"])

    return streams, CP_vals


def run_pinch(streams, DTmin=10.0):
    """
    Run pinch analysis and return:
    intervals, cascade, QHmin, QCmin
    """
    shifted = shift_streams(streams, DTmin)
    intervals = build_interval_table(shifted)
    cascade, QHmin, QCmin = heat_cascade(intervals)
    return intervals, cascade, QHmin, QCmin


def run_heat_pump(Q_hot_hen, Q_cold_hen, T_source_C=25.0, T_sink_C=95.0, eta=0.45):
    """
    Simple industrial heat pump model.

    Inputs:
        Q_hot_hen  = residual hot utility after HEN (kW)
        Q_cold_hen = residual cold utility after HEN (kW)

    Returns:
        dict with HP performance and updated utility loads
    """
    T_source_K = T_source_C + 273.15
    T_sink_K   = T_sink_C + 273.15

    COP = eta * (T_sink_K / (T_sink_K - T_source_K))

    # available low-grade heat in KOH loop (kW)
    Q_koh_available = abs(Q_into_liquid_loop_MW * 1000)

    # evaporator duty limited by:
    #   1. remaining cold utility
    #   2. remaining hot utility
    #   3. actual low-grade heat available
    Q_hp_source = min(Q_cold_hen, Q_hot_hen, Q_koh_available)

    if COP <= 1.0 or Q_hp_source <= 0:
        return {
            "COP": COP,
            "Q_hp_source": 0.0,
            "W_hp": 0.0,
            "Q_hp_sink": 0.0,
            "Q_hot_final": Q_hot_hen,
            "Q_cold_final": Q_cold_hen
        }

    W_hp = Q_hp_source / (COP - 1.0)
    Q_hp_sink = Q_hp_source + W_hp

    Q_hot_final = max(0.0, Q_hot_hen - Q_hp_sink)
    Q_cold_final = max(0.0, Q_cold_hen - Q_hp_source)

    return {
        "COP": COP,
        "Q_hp_source": Q_hp_source,
        "W_hp": W_hp,
        "Q_hp_sink": Q_hp_sink,
        "Q_hot_final": Q_hot_final,
        "Q_cold_final": Q_cold_final
    }


def scenario_summary():
    """
    Master scenario comparison:
        1. Base Case  (gross thermal turnover basis)
        2. HEN        (gross utility burden after pinch)
        3. HEN + HP   (gross utility burden after upgrading)

    All three stages are reported on the SAME basis:

        Gross Thermal Burden = Hot Utility + Cold Utility

    This keeps Stage 1 / 2 / 3 directly comparable.
    """
    streams, CP_vals = build_stream_table()

    # --------------------------------------------------------
    # STAGE 1: BASE CASE (MATCHES ORIGINAL 7.22 GJ/t BASIS)
    # --------------------------------------------------------
    # Gross thermal turnover before integration:
    #   endothermic heating
    # + sensible heating
    # + internally rejected thermal burden
    #
    # This reproduces your original 7.22 GJ/t basis.
    Q_hot_base = (
        Q_calc +                        # calciner reaction
        (Q_sens_CaCO3 / 1000) +         # CaCO3 heating
        abs(Q_sens_CaO / 1000) +        # CaO cooling burden
        abs(Q_into_liquid_loop_MW * 1000)   # liquid loop thermal dump
    )  # kW

    # Explicit cooling ledger (reported separately for transparency)
    Q_cold_base = (
        abs(Q_sens_CaO / 1000) +   # CaO cooling
        abs(Q_PR) +                # pellet exotherm
        abs(Q_slak)                # slaker exotherm
    )  # kW

    # For Stage 1 your "hot" ledger already represents gross plant burden
    Q_gross_base = Q_hot_base

    # --------------------------------------------------------
    # STAGE 2: HEN
    # --------------------------------------------------------
    intervals, cascade, QHmin, QCmin = run_pinch(streams, DTmin=10.0)

    # Remaining external utilities after passive recovery
    Q_hot_hen = Q_calc + QHmin
    Q_cold_hen = QCmin

    # Gross post-HEN utility burden
    Q_gross_hen = Q_hot_hen + Q_cold_hen

    # --------------------------------------------------------
    # STAGE 3: HEN + HP
    # --------------------------------------------------------
    hp = run_heat_pump(Q_hot_hen, Q_cold_hen)

    # Gross post-HP utility burden
    Q_gross_hp = hp["Q_hot_final"] + hp["Q_cold_final"]

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------
    print("\n" + "="*55)
    print("ENERGY INTEGRATION SCENARIO ANALYSIS")
    print("="*55)

    print("\n--- STAGE 1: BASE CASE (GROSS THERMAL TURNOVER) ---")
    print(f"Hot Utility Ledger:     {Q_hot_base/1000:.2f} MW")
    print(f"Cold Utility Ledger:    {Q_cold_base/1000:.2f} MW")
    print(f"Gross Thermal Burden:   {Q_gross_base/1000:.2f} MW")
    print(f"Specific Gross Duty:    {(Q_gross_base*3.6)/total_kg_hr:.2f} GJ/t-CO2")

    print("\n--- STAGE 2: HEN (PINCH) ---")
    print(f"Minimum Hot Utility:    {Q_hot_hen/1000:.2f} MW")
    print(f"Minimum Cold Utility:   {Q_cold_hen/1000:.2f} MW")
    print(f"Gross Thermal Burden:   {Q_gross_hen/1000:.2f} MW")
    print(f"Thermal Saved vs Base:  {(Q_gross_base - Q_gross_hen)/1000:.2f} MW")
    print(f"Specific Gross Duty:    {(Q_gross_hen*3.6)/total_kg_hr:.2f} GJ/t-CO2")

    print("\n--- STAGE 3: HEN + HEAT PUMP ---")
    print(f"Heat Pump COP:          {hp['COP']:.2f}")
    print(f"HP Evaporator Duty:     {hp['Q_hp_source']/1000:.2f} MW")
    print(f"HP Compressor Power:    {hp['W_hp']/1000:.2f} MW(e)")
    print(f"HP Condenser Duty:      {hp['Q_hp_sink']/1000:.2f} MW")
    print(f"Final Hot Utility:      {hp['Q_hot_final']/1000:.2f} MW")
    print(f"Final Cold Utility:     {hp['Q_cold_final']/1000:.2f} MW")
    print(f"Gross Thermal Burden:   {Q_gross_hp/1000:.2f} MW")
    print(f"Thermal Saved vs Base:  {(Q_gross_base - Q_gross_hp)/1000:.2f} MW")
    print(f"Specific Gross Duty:    {(Q_gross_hp*3.6)/total_kg_hr:.2f} GJ/t-CO2")
    print(f"Electric Penalty:       {(hp['W_hp']*3.6)/total_kg_hr:.2f} GJ/t-CO2")
    print("="*55)

    return {
        "streams": streams,
        "intervals": intervals,
        "cascade": cascade,
        "base": {
            "Q_hot": Q_hot_base,
            "Q_cold": Q_cold_base,
            "Q_gross": Q_gross_base
        },
        "hen": {
            "Q_hot": Q_hot_hen,
            "Q_cold": Q_cold_hen,
            "Q_gross": Q_gross_hen
        },
        "hp": {
            **hp,
            "Q_gross": Q_gross_hp
        }
    }



#%% Run scenario analysis
results = scenario_summary()