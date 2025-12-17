# -*- coding: utf-8 -*-
"""
Package requirements:
    numba
    numpy
    gekko

Created on Thu Sep 11 15:39:41 2025

Propulsion functions for Gekko Optimizations

@author: NASSAS
"""

import numpy as np
from numpy.polynomial import Polynomial
from numba import njit
inm = 0.0254

#%% Functions to parse the coefficient propeller data
def parse_coef_propeller_data(prop_name):
    """
    prop_name in the form: 16x10E, 18x12E, 12x12, etc (no PER3_ and no .dat to make it easier for new users)
    Parses the provided PER3_16x10E.dat content to extract RPM, V (m/s), Thrust (N), Torque (N-m).
    Stores in PROP_DATA as {rpm: {'V': np.array, 'Thrust': np.array, 'Torque': np.array}}
    """    
    PROP_DATA = {}

    with open(f'Databases/PropDatabase/PER3_{prop_name}.dat', 'r') as f:
        data_content = f.read()

    current_rpm = None
    in_table = False
    table_lines = []
    
    for line in data_content.splitlines():
        line = line.strip()
        if line.startswith("PROP RPM ="):
            # Extract RPM
            current_rpm = int(line.split("=")[-1].strip())
            in_table = False
            table_lines = []
        elif line.startswith("V") and "J" in line and current_rpm is not None:
            # Start of table headers
            in_table = True
        elif in_table and line and not line.startswith("(") and len(line.split()) >= 10:
            # Parse data rows (ensure it's a data line with enough columns)
            parts = line.split()
            try:
                J = float(parts[1])  # advance ratio J
                CT = float(parts[3])  # thrust coef
                CP = float(parts[4])  # power coef (can convert to CQ)
                # v_mps = v_mph * MPH_TO_MPS  # Convert to m/s
                table_lines.append((J, CT, CP))
            except (ValueError, IndexError):
                continue  # Skip malformed lines
        elif in_table and (line == "" or "PROP RPM" in line):
            # End of table for this RPM, store if data exists
            if current_rpm and table_lines:
                J_list, CT_list, CP_list = zip(*sorted(table_lines))  # Sort by V for interp1d
                PROP_DATA[current_rpm] = {
                    'J': np.array(J_list),
                    'CT': np.array(CT_list),
                    'CP': np.array(CP_list)
                }
            in_table = False
    
    # Sort RPM keys for efficient lookup
    PROP_DATA['rpm_list'] = sorted(PROP_DATA.keys())
    
    # array based datastructure where each index corresponds to rpm_values[i] (or i+1*1000 RPM)
    # and in each index there is [[V values], [Thrust values], [Torque values]] at the indices, 0, 1, 2
    numba_prop_data = []
    for RPM in PROP_DATA['rpm_list']:
        datasection = np.array([PROP_DATA[RPM]['J'], 
                                PROP_DATA[RPM]['CT'], 
                                PROP_DATA[RPM]['CP']])
        numba_prop_data.append(datasection)
        
    return(PROP_DATA, numba_prop_data)

def initialize_RPM_polynomials(PROP_DATA):
    """
    returns: rpm_values, CT_polys, CP_polys, J_DOMAINS
    
    Creates polynomial approximations for thrust and torque that are compatible with GEKKO in the form of Thrust(V) for a fixed RPM
    Uses piecewise polynomials for different RPM ranges.
    """
    
    # Extract data for polynomial fitting
    rpm_values = sorted([rpm for rpm in PROP_DATA.keys() if isinstance(rpm, int)])
    
    # Create coefficient matrices for polynomial approximation
    # We'll use separate polynomials for different velocity ranges
    CT_polys = {}
    CP_polys = {}
    
    J_DOMAINS = []
    for rpm in rpm_values:
        data = PROP_DATA[rpm]
        J_data = data['J']
        CT_data = data['CT']
        CP_data = data['CP']
                
        # Fit polynomials (degree 3-4 should be sufficient for most cases)
        CT_poly = Polynomial.fit(J_data, CT_data, deg=4)
        CT_polys[rpm] = CT_poly
        
        CP_poly = Polynomial.fit(J_data, CP_data, deg=4)
        CP_polys[rpm] = CP_poly
    
        J_DOMAINS.append(CP_poly.domain[1])
    
    J_DOMAINS = np.array(J_DOMAINS)
    rpm_values = np.array(rpm_values)
    return rpm_values, CT_polys, CP_polys, J_DOMAINS

@njit(fastmath = True)
def CTNumba(RPM, J, rpm_list, numba_prop_data):
    '''
    J: advance ratio
    
    numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    with the structure [[Jvalues], [CTvalues], [CPvalues]] for each index
    data[0] = J values, data[1] = CT, data[2] = CP
    '''
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or J < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
        
    CTs = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 -1)]
        if J > data[0].max():
            CTs.append(0.0)
            continue
        CTs.append(np.interp(J, data[0], data[1]))
        
    CTs = np.array(CTs)
    
    if len(closest_rpms) == 1:
        return CTs[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * CTs[0] + weight * CTs[1]

@njit(fastmath = True)
def CPNumba(RPM, J, rpm_list, numba_prop_data):
    '''
    J: advance ratio

    numba_prop_data is packaged so each index corresponds to (i+1)*1000 RPM 
    with the structure [[Jvalues], [CTvalues], [CPvalues]] for each index
    data[0] = J values, data[1] = CT, data[2] = CP
    '''
    if RPM < rpm_list[0] or RPM > rpm_list[-1] or J < 0:
        return 0.0
    
    idx = np.searchsorted(rpm_list, RPM)
    if idx == 0:
        closest_rpms = [rpm_list[0]]
    elif idx == len(rpm_list):
        closest_rpms = [rpm_list[-1]]
    else:
        closest_rpms = [rpm_list[idx - 1], rpm_list[idx]]
        
    CPs = []
    for rpm in closest_rpms:
        data = numba_prop_data[int(rpm/1000 - 1)]
        # NEW CODE WAS NEEDED TO BRING IT TO 0 WHEN J WAS OUTSIDE BOUNDS!!
        if J > data[0].max():
            CPs.append(0.0)
            continue
        CPs.append(np.interp(J, data[0], data[2]))
        
    CPs = np.array(CPs)
    if len(closest_rpms) == 1:
        return CPs[0]
    else:
        weight = (RPM - closest_rpms[0]) / (closest_rpms[1] - closest_rpms[0])
        return (1 - weight) * CPs[0] + weight * CPs[1]
    
#%% OLD least squares global polynomial fitting!
def getCoefs(J, RPM, data, degree = 2):
    degree_x = degree
    degree_y = degree

    # Generate grid points for evaluation
    x = J
    y = RPM
    x_grid, y_grid = np.meshgrid(x, y)

    # Flatten the grid points and the corresponding values
    X_flat = x_grid.flatten()
    Y_flat = y_grid.flatten()
    Q_flat = data.flatten()

    # Generate Vandermonde matrix
    V = np.polynomial.polynomial.polyvander2d(X_flat, Y_flat, [degree_x, degree_y])
    
    # Calculate polynomial coefficients using least squares
    coeffs, _, _, _ = np.linalg.lstsq(V, Q_flat, rcond=None)

    # Reshape the coefficients into a 2D matrix
    coeffs_matrix = np.reshape(coeffs, (degree_y + 1, degree_x + 1))
    return(coeffs_matrix)

def getValue(RPM, J, coef_matrix):
    '''Mostly shorthand, could take out if it slows down the code'''
    return(np.polynomial.polynomial.polyval2d(J, RPM, coef_matrix))

def DataCollect(numba_prop_data, rpm_values, J_DOMAINS, n = 300):
    '''Generates a coefficient matrix for CT or CP for RPM, J square! 
    (given by indx, which denotes the max RPM value to take data from)'''
    
    # alter the number of elements here to change the fit, 
    # lower means worse fit but slightly faster    
    RPMs = np.linspace(1000, rpm_values[-1], n)
    Js = np.linspace(0, J_DOMAINS.min(), n)

    # data from interpolation functions
    # print('\nGetting polynomial coefficients (~7s)')
    CT, CP = innerdatafunc(RPMs, Js, rpm_values, numba_prop_data, n)
            
    CTcoeffmatrix = getCoefs(Js, RPMs, CT, degree = 2)
    CPcoeffmatrix = getCoefs(Js, RPMs, CP, degree = 2)
    return(CTcoeffmatrix, CPcoeffmatrix)

def getCoefMats(propname):
    '''Returns, CTmat for use in getValues'''
    D = float(propname.split('x')[0])*inm
    
    PROP_DATA, numba_prop_data = parse_coef_propeller_data(propname)
    rpm_values, CT_polys, CP_polys, J_DOMAINS = initialize_RPM_polynomials(PROP_DATA)
    CTcoefmatrix, CPcoefmatrix = DataCollect(numba_prop_data, rpm_values, J_DOMAINS, n = 500)
    
    return(CTcoefmatrix, CPcoefmatrix, rpm_values, J_DOMAINS, D)

#%% for Bspline fitting of for propeller coefficients
@njit(fastmath = True)
def innerdatafunc(RPMs, Js, rpm_values, numba_prop_data, n):
    CT = np.zeros((n, n))
    CP = np.zeros((n, n))
    for i, RPM in enumerate(RPMs):
        for j, J in enumerate(Js):
            CT[i, j] = CTNumba(RPM, J, rpm_values, numba_prop_data)
            CP[i, j] = CPNumba(RPM, J, rpm_values, numba_prop_data)
    return(CT, CP)

def SplineCollect(numba_prop_data, rpm_values, J_DOMAINS, n = 100):
    ''' gathers the regularly spaced grid for spline evaluation'''
    
    # alter the number of elements here to change the fit, 
    # lower means worse fit but slightly faster    
    RPMs = np.linspace(1000, rpm_values[-1], n)
    Js = np.linspace(0, J_DOMAINS.min(), n)

    # data from interpolation functions
    # print('\nGetting polynomial coefficients (~7s)')
    CT, CP = innerdatafunc(RPMs, Js, rpm_values, numba_prop_data, n)
            
    return(RPMs, Js, CT, CP)

def getSplineData(propname):
    D = float(propname.split('x')[0])*inm
    
    PROP_DATA, numba_prop_data = parse_coef_propeller_data(propname)
    rpm_values, CT_polys, CP_polys, J_DOMAINS = initialize_RPM_polynomials(PROP_DATA)
    RPMs, Js, CT, CP = SplineCollect(numba_prop_data, rpm_values, J_DOMAINS, n = 100)
    
    return(CT, CP, RPMs, Js, rpm_values, J_DOMAINS, D)

#%% SimpleRPM model for finding T, eta, Vm, Im, Ib from design variables
def SimpleRPM(RPM, J, dT, ns, CB, Rb, rho, D, nmot, KV, I0, SOC, CP, CT, m):
    eta_c = 0.93 # large source of uncertainty
    
    n =     RPM/60
    Q =     rho*(n**2)*(D**5)*(CP/(2*np.pi))
    Im =    Q*KV*(np.pi/30) + I0
    Ib =    (Im*nmot)/(eta_c)
    Voc =   VocFuncGK(SOC, m)
    Vb =    ns*(Voc) - Ib*Rb
    Vm =    dT*Vb

    Pout =  rho*(n**3)*(D**5)*CP
    Pin_m = Vm*Im    
    eta_p = (CT*J)/CP
    eta_m = Pout/Pin_m
    Pin_c = (Ib/nmot)*Vb
    eta_b = 1.0 - ((Ib**2)*Rb)/(nmot*Pin_c + (Ib**2)*Rb)
    eta_drive = ((eta_p*eta_m*eta_c)/nmot)*eta_b
    T =     nmot*rho*(n**2)*(D**4)*CT
    
    return(eta_drive, T, Vm, Im, Ib)

#%% Battery voltage functions for lipos
def VocFunc(SOC):
    '''
    Determines the battery Voltage (Voc) as a function of State of Charge (SOC) given from 0-1
    
    Main equation from Chen 2006
    https://rincon-mora.gatech.edu/publicat/jrnls/tec05_batt_mdl.pdf
    
    Alternative from Jeong 2020
    Voc = 1.7*(SOC**3) - 2.1*(SOC**2) + 1.2*SOC + 3.4
    https://www.researchgate.net/publication/347270768_Improvement_of_Electric_Propulsion_System_Model_for_Performance_Analysis_of_Large-Size_Multicopter_UAVs
    
    NOTE:
    Jeong 2020 also presents resistance as a function of cell energy:
    Rbatt.cell = 21.0*(ebatt.cell)**-0.8056
    
    '''
    return(3.685 - 1.031 * np.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3)

def VocFuncGK(SOC, m):
    '''
    COMPATIBLE WITH GEKKO 
    
    Determines the battery Voltage (Voc) as a function of State of Charge (SOC) given from 0-1
    
    Main equation from Chen 2006
    https://rincon-mora.gatech.edu/publicat/jrnls/tec05_batt_mdl.pdf
    
    Alternative from Jeong 2020
    Voc = 1.7*(SOC**3) - 2.1*(SOC**2) + 1.2*SOC + 3.4
    https://www.researchgate.net/publication/347270768_Improvement_of_Electric_Propulsion_System_Model_for_Performance_Analysis_of_Large-Size_Multicopter_UAVs
    
    NOTE:
    Jeong 2020 also presents resistance as a function of cell energy:
    Rbatt.cell = 21.0*(ebatt.cell)**-0.8056
    
    SOC FINDER FOR MANUAL USAGE:
    https://www.desmos.com/calculator/vjfkdfgcga
    
    '''
    return(3.685 - 1.031 * m.exp(-35 * SOC) + 0.2156 * SOC - 0.1178 * SOC**2 + 0.3201 * SOC**3)


#%% printing function for Gekko (GK) variables
def SM(variable, units=''):
    '''takes in a GK variable with units of metric, 
    returns a string with the rounded value converted to imperial'''
    ftm = 0.3048 #multiply ft by this to get m
    ft2m2 = 0.3048**2 #for ft2 to m2
    lbfN = 4.44822 #multiply lbf by this to get N
    if units == 'ft/s' or units == 'ft':
        val = round(variable.value[0]/ftm, 4)
        print(f'{variable.name:25}= {val:0.6} {units}')
    elif units == 'lbf':
        val = round(variable.value[0]/lbfN, 4)
        print(f'{variable.name:25}= {val:0.6} {units}')
    elif units == 'ft2':
        val = round(variable.value[0]/ft2m2, 4)
        print(f'{variable.name:25}= {val:0.6} {units}')
    elif units == 's' or units == 'deg/s':
        val = round(variable.value[0], 4)
        print(f'{variable.name:25}= {val:0.6} {units}')
    else:
        val = round(variable.value[0], 4)
        print(f'{variable.name:25}= {val:0.6}')  

