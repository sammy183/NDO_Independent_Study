# -*- coding: utf-8 -*-
"""
Package requirements: 
    gekko (https://gekko.readthedocs.io/en/latest/)
    aerosandbox (https://github.com/peterdsharpe/AeroSandbox)

NDO Independant Study

@author: NASSAS
"""

from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import aerosandbox as asb

from propulsionsOpt import getSplineData, SimpleRPM, VocFuncGK
from aerodynamicsOpt import CDs 

inm = 0.0254    
ftm = 0.3048
lbfN = 4.44822
gN = 0.00980665 # gram-force to N

#%% Optimization
def RunDBFOpt(M2propname, M3propname, fixnpax = False, folder = False, verbose = False, printout = True, solver = 1):
    CT_M2, CP_M2, RPMs_M2, JsM2, rpm_valsM2, J_domainsM2, DM2 = getSplineData(M2propname)
    CT_M3, CP_M3, RPMs_M3, JsM3, rpm_valsM3, J_domainsM3, DM3 = getSplineData(M3propname)

    ############# ASSUMPTIONS/V1 CONSTANTS ################
    # standard conditions
    g = 9.807
    rho = 1.191
    mu = 1.81e-5

    # based on propulsion regression
    Wmotor = 400    #g
    Wbat = 554      #g
    Rm = 0.03     # motor resistance
    maxds = 0.8 # maximum battery discharge 

    # limits on RPM and J from the propeller data
    RPMlimM2 = float(rpm_valsM2[-1])
    JlimM2 = J_domainsM2.min()
    RPMlimM3 = float(rpm_valsM3[-1])
    JlimM3 = J_domainsM3.min()

    #%%################ MAIN OPTIMIZATION ###########################
    if printout:
        print('\nStarting Optimization')
        
    # initialize gekko model
    m = GEKKO(remote = False)
    
    m.Const(0.0, name = f'M2 Prop: {M2propname}') # to have propname in the output
    m.Const(0.0, name = f'M3 Prop: {M3propname}') # to have propname in the output
    ##############################################################################
    ############### VARIABLE DEFINITION ##########################################
    ##############################################################################
    # wing vars
    b =     m.Var(4.0*ftm, lb = 3*ftm, ub = 5*ftm, name = 'wingspan')
    c_r =   m.Var(0.8*ftm, lb = 0.2*ftm, name = 'root chord')
    taper = m.Var(0.43, lb = 0.3, ub = 1.0, name = 'taper') 
    sweep = m.Var(0.0, lb = 0.0, ub = 15, name = 'sweep')
    t =     m.Var(0.12, lb = 0.1, name = 'thickness')
    
    # fusevars
    w =     m.Var(1.0*ftm, lb = 5*inm, name = 'fuselage_width') 
    h =     m.Const(0.5*ftm, name = 'fuselage_height')
    
    # Aero and performance
    CLcM2 = m.Var(0.8, lb = 0.05, ub = 0.9, name = 'CL M2 Cruise')
    CLcM3 = m.Var(0.8, lb = 0.05, ub = 0.9, name = 'CL M3 Cruise')
    CLtM2 = m.Var(0.7, lb = 0.5, ub = 2.0, name = 'CL M2 Turn')
    CLtM3 = m.Var(0.7, lb = 0.5, ub = 2.0, name = 'CL M3 Turn')
    
    minU = 13.5 # 30 mph minimum to maintain gust performance
    UcM2 = m.Var(30, lb = minU, ub = 45, name = 'Cruise Velocity M2') 
    UcM3 = m.Var(30, lb = minU, ub = 45, name = 'Cruise Velocity M3')
    UtM2 = m.Var(20, lb = minU, ub = 45, name = 'Turn Velocity M2')
    UtM3 = m.Var(20, lb = minU, ub = 45, name = 'Turn Velocity M3')

    nM2 = m.Var(7, lb = 1.5, ub = 12, name = 'Load Factor M2') # keep around 300 n*Wtot max
    nM3 = m.Var(7, lb = 1.5, ub = 12, name = 'Load Factor M3')
    
    # Propulsion variables
    nmot = m.Var(1, lb = 1, integer = True, name = 'number of motors')
    KV =   m.Var(300, lb = 100, name = 'Motor KV')
    Pmax = m.Var(3000, lb = 400, ub = 6000, name = 'Motor Max Power')
    CBM2 = m.Var(3300, name = 'Battery Capacity M2')
    CBM3 = m.Var(3300, name = 'Battery Capacity M3')
    nsM2 = m.Var(8, lb = 4, ub = 12, integer = True, name = 'Battery Cells M2')
    nsM3 = m.Var(8, lb = 4, ub = 12, integer = True, name = 'Battery Cells M3')
    EM2 = 3.7*nsM2*(CBM2/1000)
    EM3 = 3.7*nsM3*(CBM3/1000)
    
    dTcruiseM2 = m.Var(0.5, lb = 0.3, ub = 1.0, name = 'Throttle M2 Cruise')
    dTcruiseM3 = m.Var(0.5, lb = 0.3, ub = 1.0, name = 'Throttle M3 Cruise')
    dTturnM2 =   m.Var(0.5, lb = 0.3, ub = 1.0, name = 'Throttle M2 Turn')
    dTturnM3 =   m.Var(0.5, lb = 0.3, ub = 1.0, name = 'Throttle M3 Turn')
    
    # PROP MODEL VARIABLES (NOTE: the prop model must be solved for both cruise, turn on both M2, M3)
    RPMcM2 = m.Var((rpm_valsM2[0]+rpm_valsM2[-1])/2, lb = 1000, ub = RPMlimM2, name = 'RPM M2 Cruise')
    RPMcM3 = m.Var((rpm_valsM3[0]+rpm_valsM3[-1])/2, lb = 1000, ub = RPMlimM3, name = 'RPM M3 Cruise')
    RPMtM2 = m.Var((rpm_valsM2[0]+rpm_valsM2[-1])/2, lb = 1000, ub = RPMlimM2, name = 'RPM M2 Turn')
    RPMtM3 = m.Var((rpm_valsM3[0]+rpm_valsM3[-1])/2, lb = 1000, ub = RPMlimM3, name = 'RPM M3 Turn')
    
    # now we need to define the advance ratios as variables themselves, then constrain them to the velocities
    JcM2 = m.Var((JsM2.min() + JsM2.max())/2, lb = JsM2.min(), ub = JsM2.max(), name = 'JcM2')
    JtM2 = m.Var((JsM2.min() + JsM2.max())/2, lb = JsM2.min(), ub = JsM2.max(), name = 'JtM2')
    JcM3 = m.Var((JsM3.min() + JsM3.max())/2, lb = JsM3.min(), ub = JsM3.max(), name = 'JcM3')
    JtM3 = m.Var((JsM3.min() + JsM3.max())/2, lb = JsM3.min(), ub = JsM3.max(), name = 'JtM3')
    
    # slack for CTs, CPs 
    CTcM2 = m.Var(0.01, lb = 0.0)
    CTtM2 = m.Var(0.01, lb = 0.0)
    CTcM3 = m.Var(0.01, lb = 0.0)
    CTtM3 = m.Var(0.01, lb = 0.0)
    
    CPcM2 = m.Var(0.01, lb = 0.0)
    CPtM2 = m.Var(0.01, lb = 0.0)
    CPcM3 = m.Var(0.01, lb = 0.0) 
    CPtM3 = m.Var(0.01, lb = 0.0)
    
    degs = 5
    # CT fits
    m.bspline(RPMcM2, JcM2, CTcM2, RPMs_M2, JsM2, CT_M2, data = True, kx = degs, ky = degs)
    m.bspline(RPMtM2, JtM2, CTtM2, RPMs_M2, JsM2, CT_M2, data = True, kx = degs, ky = degs)
    m.bspline(RPMcM3, JcM3, CTcM3, RPMs_M3, JsM3, CT_M3, data = True, kx = degs, ky = degs)
    m.bspline(RPMtM3, JtM3, CTtM3, RPMs_M3, JsM3, CT_M3, data = True, kx = degs, ky = degs)
    
    # CP fits
    m.bspline(RPMcM2, JcM2, CPcM2, RPMs_M2, JsM2, CP_M2, data = True, kx = degs, ky = degs)
    m.bspline(RPMtM2, JtM2, CPtM2, RPMs_M2, JsM2, CP_M2, data = True, kx = degs, ky = degs)
    m.bspline(RPMcM3, JcM3, CPcM3, RPMs_M3, JsM3, CP_M3, data = True, kx = degs, ky = degs)
    m.bspline(RPMtM3, JtM3, CPtM3, RPMs_M3, JsM3, CP_M3, data = True, kx = degs, ky = degs)
    
    # MISSION VARIABLES
    npaxcols = m.Var(1, lb = 1, integer = True, name = 'Number of Pax Columns')
    npaxrows = m.Var(3, lb = 1, integer = True, name = 'Number of Pax Rows')

    nperstack = m.Var(1, lb = 1, integer = True, name = 'Pucks Per Stack')
    nstacks =   m.Var(1, lb = 1, integer = True, name = 'Number of Puck Stacks')

    npax = m.Intermediate(npaxcols*npaxrows, name = 'npax')
    ncargo = m.Intermediate(nperstack*nstacks, name = 'ncargo')
    
    if fixnpax != False:
        m.Equation(npax == fixnpax)
    
    Lbanner =   m.Var(150, lb = 0.254, name = 'Banner Length')

    nlapsM2 =       m.Var(7, lb = 1, integer = True, name = 'nlapsM2')
    nlapsM3 =       m.Var(7, lb = 1, integer = True, name = 'nlapsM3')    
    
    #%%##############################################################################
    ########################## CONNECTIONS #######################################
    ##############################################################################
    # initial propulsions
    I0 = m.Intermediate(0.00018790*Pmax + 1.56205099, name = 'I0')
    RbM2 = m.Intermediate(0.00296778*nsM2 - 0.00506889, name = 'RbM2')
    RbM3 = m.Intermediate(0.00296778*nsM3 - 0.00506889, name = 'RbM3')

    #### GEOMETRY
    Splan =     m.Intermediate(0.5*(c_r*b*(1 + taper)), name = 'Planform Wing Area') #m2, planform, DOES NOT account for the fuselage taking up area too now!
    AR =        m.Intermediate((b**2)/Splan, name = 'AR')
    Sbanner =   m.Intermediate(0.2*(Lbanner**2), name = 'Planform Banner Area') #m2
    Sw =        m.Intermediate(Splan - c_r*w, name = 'Wing Area uncovered')
        
    #### WEIGHTS
    def Wempty_calc(Wpay):
        '''Wpay input in N, converted to lbs for the equation, then output as N'''
        return((0.2144405633683760*(Wpay/lbfN) + 4.8014899539466200)*lbfN)
    
    # propulsion weight
    Wprop = Wmotor*gN + Wbat*gN + 0.06875*lbfN # last constant is the weight of a 16x10E propeller in lbs. I assume most props will not alter this much
    
    # M2
    Wpax = (0.04375*lbfN)*npax 
    Wcargo = (0.375*lbfN)*ncargo
    WpayM2 = Wpax + Wcargo

    # M3
    Wbannermech = 1.0*lbfN # assume 1 lb for now
    Wbanner = (0.23275)*Sbanner #g/m2, converted to N/m2 ASSUMES PICNIC BLANKET LIKE MATERIAL!!!
    WpayM3 = Wbannermech + Wbanner 
    
    # Empty weight defined by the maximum payload weight of either M2 or M3
    if solver == 1:
        Wpaymax = m.max3(WpayM2, WpayM3) # for discrete optimization
    elif solver == 3:
        Wpaymax = m.max2(WpayM2, WpayM3) # for continuous optimization
    Wempty = m.Intermediate(Wempty_calc(Wpaymax), name = 'Wempty')
    WtotM2 = m.Intermediate(WpayM2 + Wprop + Wempty, name = 'Wtot M2')
    WtotM3 = m.Intermediate(WpayM3 + Wprop + Wempty, name = 'Wtot M3')
    
    ### FUSELAGE SIZING
    # width of the fuselage must be greater than the puck or pax payload
    m.Equation([w >= npaxcols*(2.7*inm) + 1*inm,
                w >= nperstack*inm + 2*inm,
                ]) 
    # assume stacks are positioned so each one is 3 in long, each passenger row is 2.5 in, extra 10 in for battery + motor, upsweep
    l = m.Intermediate(nstacks*3*inm + npaxrows*2.5*inm + 10*inm, name = 'Fuselage Length') 
    Sfusepay = m.Intermediate(2*(l*w) + 2*(l*h), name = 'Fuselage Payload Surface Area')
    
    
    ### AERODYNAMICS
    CDcM2 = CDs(UcM2, CLcM2, rho, mu, l, w, h, Sfusepay, c_r, Sw, t, AR, taper, sweep, m)
    CDtM2 = CDs(UtM2, CLtM2, rho, mu, l, w, h, Sfusepay, c_r, Sw, t, AR, taper, sweep, m)
    
    # for M3 we have additional banner drag!
    a_banner = 0.0364 # army study
    # a_banner = 0.0409 # USC 2020
    # a_banner = 0.0238 # our data (dubious)
    CD0ban = m.Intermediate(a_banner*(Sbanner/Sw), name = 'CD Banner')
    CDcM3 = CDs(UcM3, CLcM3, rho, mu, l, w, h, Sfusepay, c_r, Sw, t, AR, taper, sweep, m) + CD0ban
    CDtM3 = CDs(UtM3, CLtM3, rho, mu, l, w, h, Sfusepay, c_r, Sw, t, AR, taper, sweep, m) + CD0ban

    # redefine intermediates for calling after optimization
    CDcM2 = m.Intermediate(CDcM2, name = 'CD M2 Cruise')
    CDcM3 = m.Intermediate(CDcM3, name = 'CD M3 Cruise')
    CDtM2 = m.Intermediate(CDtM2, name = 'CD M2 Turn')
    CDtM3 = m.Intermediate(CDtM3, name = 'CD M3 Turn')
    
    qcM2 = 0.5*rho*(UcM2**2)
    qcM3 = 0.5*rho*(UcM3**2)
    qtM2 = 0.5*rho*(UtM2**2)
    qtM3 = 0.5*rho*(UtM3**2)
    
    LcM2 = m.Intermediate(qcM2*CLcM2*Sw, name = 'Lift M2 Cruise')
    LcM3 = m.Intermediate(qcM3*CLcM3*Sw, name = 'Lift M3 Cruise')
    LtM2 = m.Intermediate(qtM2*CLtM2*Sw, name = 'Lift M2 Turn')
    LtM3 = m.Intermediate(qtM3*CLtM3*Sw, name = 'Lift M3 Turn') 
    
    DcM2 = m.Intermediate(qcM2*CDcM2*Sw, name = 'Drag M2 Cruise') 
    DcM3 = m.Intermediate(qcM3*CDcM3*Sw, name = 'Drag M3 Cruise') 
    DtM2 = m.Intermediate(qtM2*CDtM2*Sw, name = 'Drag M2 Turn')
    DtM3 = m.Intermediate(qtM3*CDtM3*Sw, name = 'Drag M3 Turn') 
    
    #### ALL PROPULSIONS    
    eta_drive_cM2, TcM2, VmcM2, ImcM2, IbcM2 = SimpleRPM(RPMcM2, JcM2, dTcruiseM2, nsM2, CBM2, RbM2, rho, DM2, nmot, KV, I0, 1.1-maxds, CPcM2, CTcM2, m)
    eta_drive_cM3, TcM3, VmcM3, ImcM3, IbcM3 = SimpleRPM(RPMcM3, JcM3, dTcruiseM3, nsM3, CBM3, RbM3, rho, DM3, nmot, KV, I0, 1.1-maxds, CPcM3, CTcM3, m)
    eta_drive_tM2, TtM2, VmtM2, ImtM2, IbtM2 = SimpleRPM(RPMtM2, JtM2, dTturnM2, nsM2, CBM2, RbM2, rho, DM2, nmot, KV, I0, 1.1-maxds, CPtM2, CTtM2, m)
    eta_drive_tM3, TtM3, VmtM3, ImtM3, IbtM3 = SimpleRPM(RPMtM3, JtM3, dTturnM3, nsM3, CBM3, RbM3, rho, DM3, nmot, KV, I0, 1.1-maxds, CPtM3, CTtM3, m)
        
    TcM2 = m.Intermediate(TcM2, name = 'Thrust M2 Cruise')
    TcM3 = m.Intermediate(TcM3, name = 'Thrust M3 Cruise')
    TtM2 = m.Intermediate(TtM2, name = 'Thrust M2 Turn')
    TtM3 = m.Intermediate(TtM3, name = 'Thrust M3 Turn')
    
    eta_drive_cM2 = m.Intermediate(eta_drive_cM2, name = 'Drive Efficiency M2 Cruise')
    eta_drive_cM3 = m.Intermediate(eta_drive_cM3, name = 'Drive Efficiency M3 Cruise')
    eta_drive_tM2 = m.Intermediate(eta_drive_tM2, name = 'Drive Efficiency M2 Turn')
    eta_drive_tM3 = m.Intermediate(eta_drive_tM3, name = 'Drive Efficiency M3 Turn')
    
    IbcM2 = m.Intermediate(IbcM2, name = 'IbcM2')
    IbcM3 = m.Intermediate(IbcM3, name = 'IbcM3')
    IbtM2 = m.Intermediate(IbtM2, name = 'IbtM2')
    IbtM3 = m.Intermediate(IbtM3, name = 'IbtM3')
    
    ### MISSION PERFORMANCE
    # M2 range
    dlapM2 =        609.6 + 4*np.pi*((UtM2**2)/(g*m.sqrt(nM2**2 - 1)))

    tcM2 =          m.Intermediate(609.6/UcM2, name = 'Cruise Time M2 Lap') # time for cruise in one lap
    turnrateM2 =    m.Intermediate(((g*m.sqrt(nM2**2 - 1))/UtM2)*180/np.pi, name = 'Turnrate M2 deg/s')
    ttM2 =          m.Intermediate(720/turnrateM2, name = 'Turn Time M2 Lap') # time for turn in one lap
    tlapM2 =        m.Intermediate(tcM2 + ttM2, name = 'Laptime M2') # time for one lap
    LDetaM2 =       m.Intermediate((LcM2/DcM2)*(tcM2/tlapM2)*eta_drive_cM2 + (LtM2/DtM2)*(ttM2/tlapM2)*eta_drive_tM2, name = 'L/D * eta_drive M2')

    RangeM2 =       m.Intermediate(1000*3.6*LDetaM2*(EM2*maxds)/WtotM2, name = 'Range M2') # assume only 90% of battery energy is used bc we don't go till 100% depletion
    tM2 =           m.Intermediate(nlapsM2*tlapM2, name = 'Time M2')

    # M3 range
    dlapM3 =        609.6 + 4*np.pi*((UtM3**2)/(g*m.sqrt(nM3**2 - 1)))

    tcM3 =          m.Intermediate(609.6/UcM3, name = 'Cruise Time M3 Lap') # time for cruise in one lap
    turnrateM3 =    m.Intermediate(((g*m.sqrt(nM3**2 - 1))/UtM3)*180/np.pi, name = 'Turnrate M3 deg/s')
    ttM3 =          m.Intermediate(720/turnrateM3, name = 'Turn Time M3 Lap') # time for turn in one lap
    tlapM3 =        m.Intermediate(tcM3 + ttM3, name = 'Laptime M3') # time for one lap
    LDetaM3 =       m.Intermediate((LcM3/DcM3)*(tcM3/tlapM3)*eta_drive_cM3 + (LtM3/DtM3)*(ttM3/tlapM3)*eta_drive_tM3, name = 'L/D * eta_drive M3')

    RangeM3 =       m.Intermediate(1000*3.6*LDetaM3*(EM3*maxds)/WtotM3, name = 'Range M3')
    tM3 =           m.Intermediate(nlapsM3*tlapM3, name = 'Time M3')

    m.Equation([nlapsM2 <= RangeM2/dlapM2,  # constraint #1 on nlaps based on range
                nlapsM3 <= RangeM3/dlapM3, 
                nlapsM2 <= 285/tlapM2,      # constraint #2 on nlaps based on mission time (300s with a Factor of Safety)
                nlapsM3 <= 285/tlapM3, 
                nlapsM2 <= (3.6*CBM2*maxds)/(IbcM2*tcM2 + IbtM2*ttM2), # constraint #3 based on battery discharge
                nlapsM3 <= (3.6*CBM3*maxds)/(IbcM3*tcM3 + IbtM3*ttM3),
                ])

    # GMF
    kp = 1.25 # now assume 0.8 passengers per s
    kc = 1.0
    tLoad = kp*npax + kc*nstacks
    tRest = (kp/2) * npax  + 3 
    tGM = m.Intermediate(tLoad + tRest, name = 'GM Time')
        
    ##################### Objective Functions #####################
    maxM2 = 4263.74 #3500 gotten from setting the objective to M2 alone and trying a variety of props (0.66*5000 was the best result)
    maxM3 = 1445 # was the maximum for a variety of props
    tminGM = 9.625 # theoeretical minimum for given assumptions

    EF = EM2/100
    RAC = 0.05*(b/ftm) + 0.75
        
    M1 = 1.0 
    M2 = m.Intermediate(1 + (6*npax + 10*ncargo + nlapsM2*((2-0.5*EF)*npax + (8-2*EF)*ncargo - 10*EF))/maxM2, name = 'M2 Score')
    M3 = m.Intermediate(2 + ((nlapsM3*Lbanner/inm)/RAC)/maxM3, name = 'M3 Score')
    GM = m.Intermediate(tminGM/tGM, name = 'GM Score')
    
    Tot = m.Intermediate(M1 + M2 + M3 + GM, name = 'Total Score')
    
    ##############################################################################
    ########################## CONSTRAINTS #######################################
    ##############################################################################    
    m.Equation([# CRUISE AND TURN CONSTRAINTS
                LcM2 == WtotM2, 
                TcM2 == DcM2, 
                
                LcM3 == WtotM3,
                TcM3 == DcM3, 
                
                LtM2 == nM2*WtotM2, 
                TtM2 == DtM2, 
                
                LtM3 == nM3*WtotM3,
                TtM3 == DtM3,
                
                # PROPULSION EQNS
                RPMcM2 == KV*(VmcM2 - ImcM2*Rm),
                VmcM2*ImcM2 <= Pmax,
                
                RPMcM3 == KV*(VmcM3 - ImcM3*Rm),
                VmcM3*ImcM3 <= Pmax,
                
                RPMtM2 == KV*(VmtM2 - ImtM2*Rm),
                VmtM2*ImtM2 <= Pmax,
                
                RPMtM3 == KV*(VmtM3 - ImtM3*Rm),
                VmtM3*ImtM3 <= Pmax,
                
                # so data stays within the propeller data boundary            
                JcM2 <= JlimM2,            
                JcM3 <= JlimM3,
                JtM2 <= JlimM2, 
                JtM3 <= JlimM3,
                
                # COMPETITION REQUIRED
                # Energy limit
                EM2 <= 100,
                EM3 <= 100,
                
                # Current limit (100A by comp rules, 115A as acceptable risk)
                IbcM2 <= 115, 
                IbcM3 <= 115, 
                IbtM2 <= 115, 
                IbtM3 <= 115, 

                tM2 <= 285,  #300*FS,
                tM3 <= 285,  #300*FS, # trying to adjust so it somewhat matches with acceleration model
                
                npax >= ncargo*3, 
                tGM <= 240,
                l <= 8*ftm,
                ])
    
    # running
    m.Maximize(Tot)
    m.options.SOLVER = solver
    m.options.DIAGLEVEL = 2
    m.options.MAX_ITER = 300
    if folder:
        m.open_folder()
    m.solve(disp=verbose)
    
    if printout:
        print(f'\nM2 Prop: {M2propname}')
        print(f'M3 Prop: {M3propname}')
        print(f'{"GMscore":25}: {GM.value[0]}')
        print(f'{"M2score":25}: {M2.value[0]}')
        print(f'{"M3score":25}: {M3.value[0]}')
        print(f'{"Total Score":25}: {Tot.value[0]}')    
        print(f'{"nlaps2":25}: {nlapsM2.value[0]}')
        print(f'{"nlaps3":25}: {nlapsM3.value[0]}')
        print(f'{"npax":25}: {npax.value[0]}')
        print(f'{"(row x col)":25}: {npaxrows.value, npaxcols.value}')
        print(f'{"ncargo":25}: {ncargo.value}')
        print(f'{"(stacks x puck per stack)":25}: {nstacks.value, nperstack.value}')
        print(f'{"Banner Length":25}: {Lbanner.value[0]/inm:.5f} inches')
        print(f'{"Wtot M2":25}: {WtotM2.value[0]/lbfN:.2f} lbf')
        print(f'{"Wtot M3":25}: {WtotM3.value[0]/lbfN:.2f} lbf\n')
        print(f'{"nmot":25}: {nmot.value[0]:.2f}')
        print(f'{"KV":25}: {KV.value[0]:.2f}')
        print(f'{"Pmax":25}: {Pmax.value[0]:.2f}')
        print(f'{"nsM2":25}: {nsM2.value[0]:.2f}')
        print(f'{"nsM3":25}: {nsM3.value[0]:.2f}')
        print(f'{"CBM2":25}: {CBM2.value[0]:.2f}')
        print(f'{"CBM3":25}: {CBM3.value[0]:.2f}')
        print(f'{"dTcruiseM2":25}: {dTcruiseM2.value[0]:.2f}')
        print(f'{"dTturnM2":25}: {dTturnM2.value[0]:.2f}')
        print(f'{"dTcruiseM3":25}: {dTcruiseM3.value[0]:.2f}')
        print(f'{"dTturnM3":25}: {dTturnM3.value[0]:.2f}')
        
    
    #%% Using aerosandbox to draw the aircraft output
    def PlotOptimizedAircraft(Sw, b, c_r, taper, wing_airfoil, wfuse, hfuse, lfuse):
        '''All inputs in metric, taper as taper ratio, wing_airfoil as asb.Airfoil obj'''
        c_tip = taper*c_r
        curvy = 10
        air = asb.Airplane(
            name="Sammy's foolishness",
            xyz_ref = [0, 0, 0],  # CG location, I made it up
            c_ref = c_r,
            b_ref = b,
            s_ref = Sw,
            wings=[
                asb.Wing( # NOTE: I positioned it halfway along the fuselage just for now
                    name="Main Wing",
                    symmetric=True,  # Should this wing be mirrored across the XZ plane?
                    xsecs=[  # The wing's cross ("X") sections
                        asb.WingXSec(  # Root
                            xyz_le=[2*lfuse/5, 0.0, 0.0],  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
                            chord=c_r,
                            twist=0.0,  # degrees
                            airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
                        ),
                        asb.WingXSec(  # Tip
                            xyz_le=[2*lfuse/5, b/2, 0.0],
                            chord = c_tip,
                            twist=0.0,
                            airfoil=wing_airfoil,
                        ),
                    ])],
            fuselages = [
                asb.Fuselage(
                    name="Fuselage",
                    width = wfuse,
                    height = hfuse,
                    shape = curvy,
                    xsecs=[
                        asb.FuselageXSec(xyz_c=[0.0, 0.0, 0.0],
                                         width = wfuse,
                                         height = hfuse,
                                         shape = curvy),
                        asb.FuselageXSec(xyz_c=[lfuse, 0.0, 0.0], 
                                         width = wfuse,
                                         height = hfuse,
                                         shape = curvy)
                        ])]
            )
        
        print('\nWing Sizing:')
        print(f'{"Span (b)":25}: {b/ftm:.3f} ft')
        print(f'{"Root Chord (c_r)":25}: {c_r/ftm:.3f} ft')
        print(f'{"Taper Ratio":25}: {taper}')
        print(f'{"Wing Area (Sw)":25}: {Sw/ftm/ftm:.3f} ft2')
        print(f'{"Aspect Ratio (AR)":25}: {AR.value[0]:.3f}')
        
        print('\nFuselage Sizing:')
        print(f'{"Width":25}: {wfuse/ftm:.3f} ft')
        print(f'{"Length":25}: {lfuse/ftm:.3f} ft')
        print(f'{"Height":25}: {hfuse/ftm:.3f} ft')
        
        return(air)
    if printout:
        aircraft = PlotOptimizedAircraft(float(Sw.value[0]), float(b.value[0]), float(c_r.value[0]), float(taper.value[0]), asb.Airfoil('clarky'), 
                                      float(w.value[0]), h.value, float(l.value[0]))
        aircraft.draw_three_view()     
        plt.rcdefaults()
            
    Lbannerout = Lbanner.value[0]
    npaxout = npax.value[0]
    ncargoout = ncargo.value[0]
    nlaps2out = nlapsM2.value[0]
    nlaps3out = nlapsM3.value[0]
    Swout = Sw.value[0]
    ARout = AR.value[0]
    WtotM2out = WtotM2.value[0]
    WtotM3out = WtotM3.value[0]
    lout = l.value[0]
    Wemptyout = Wempty.value[0]
    KVout = KV.value[0]
    Pmaxout = Pmax.value[0]
    bout = b.value[0]
    
    # M2 (0), M3 (1), GM (2), Tot (3), npax (4), ncargo (5), nlapsM2 (6), nlapsM3 (7), 
    # Lbanner (8), Sw (9), AR (10), Wtot M2 (11), Wtot M3 (12), lfuse (13), Wempty (14), KV (15), Pmax (16)
    out = [M2.value[0], M3.value[0], GM.value[0], Tot.value[0], npaxout, ncargoout, nlaps2out, nlaps3out, 
           Lbannerout, Swout, ARout, WtotM2out, WtotM3out, lout, Wemptyout, KVout, Pmaxout, bout]
    return(out)
