"""
Development space for functions
"""


def calc_convMT(G, V=V_col, Xs=Xs_col, u_l=u_l):
    mdot_g=G*28.96e-3 #mol/time * MW (kg/mol) of air --> mass flow rate of gas
    mdot_l=u_l*Xs*1000 # fluid velocity * area = volumetric flow rate * density of water
    c=1.3
    n=0.6
    rhs=c*(mdot_l/mdot_g)**(-n)
    
    return (rhs*mdot_l+Xs)/V
    
    
