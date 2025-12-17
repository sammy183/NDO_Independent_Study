# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 12:44:39 2025

Aerodynamics functions

@author: NASSAS
"""
import numpy as np

#%% Functions
def Oswald(U, AR, taper, sweep, t, m):
    '''
    Oswald efficiency calculation via the howe method
    Has sea level assumption built in with a = 343 m/s
    '''
    sweep = sweep*(np.pi/180) # convert to rad
    f_taper = 0.005*(1 + 1.5*((taper - 0.6)**2))
    M = U/343 #V input in m/s! Also assumes SSL
    Ne = 0 #number of engines located above the top surface of the wing
    e = 1/((1 + 0.12*(M**2))*(1 + (0.142 + f_taper*AR*((10*t)**0.33))/(m.cos(sweep)**2) + (0.1*(3*Ne+1))/((4 + AR)**0.8)))
    return(e)

# drag buildup functions
def CD0fuselage(rho, mu, U, l, w, h, Sfusepay, Sw, m):
    '''
    Parasitic drag buildup for a general square fuselage with rounded corners
    '''
    Re = (rho*U*l)/mu
    Cf = 0.455/((m.log10(Re))**2.58) 
    Amax = w*h
    f = l/m.sqrt((4/np.pi)*Amax)
    FF = 0.9 + 5/(f**1.5) + f/400
    FF = (FF-1)*0.7 + 1 #(reduce 30% to form factor greater than 1 for square fuselage with rounded corners(raymer pg 423))
    CD0f = (Cf*FF*Sfusepay)/Sw + 0.004 # 0.004 for nose + boom
    return(CD0f)

def CD0wing(rho, mu, U, c, t, Sw, m):
    '''
    Parasitic drag buildup for wing
    '''
    FFwing =    1+(2*t)+(60*(t**4))
    Swet_wing = 2*(1+((t)**2))*Sw # approximate! from
    Re = (rho*U*c)/mu
    CfW = 0.455/((m.log10(Re))**2.58) # NO CUTOFF (not good for foam!)
    CD0w = (FFwing*Swet_wing*CfW)/Sw
    return(CD0w)

def CDs(U, CL, rho, mu, l, w, h, Sfusepay, c, Sw, t, AR, taper, sweep, m):
    '''
    Drag estimation from aircraft geometry and CL
    '''
    # #via raymer eqns
    CD0f = CD0fuselage(rho, mu, U, l, w, h, Sfusepay, Sw, m)
    CD0w = CD0wing(rho, mu, U, c, t, Sw, m)
    CD0extra = 0.009 # 0.006 for landing gear, 0.003 for tails (VERY APPROX)
    CD0 = CD0f + CD0w + CD0extra
    
    e = Oswald(U, AR, taper, sweep, t, m)
    CDi = (CL**2)/(np.pi*AR*e)
    
    CD = CD0 + CDi
    return(CD)