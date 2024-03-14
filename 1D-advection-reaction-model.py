import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# initial mineral proportion
cal_frac = 0.00
arag_frac = 1

# fractionation factors
alpha_Ca = 1
alpha_Mg = 0.998
alpha_Sr = 1
alpha_Li = 0.999

# length scale
M = 0.03        # Stoichiometric ratio of Mg to C
v = 0.01        # advection rate (m/yr)
R = 2e-5        # reaction rate (yr-1)
time = 1e6      # time scale for modeling (yr), adjust the value of t to ensure a complete diagenesis
boxnum = 50     # box numbers (approximating sediment depth)

# molar mass
Ca_molar = 40.078
Mg_molar = 24.305
Sr_molar = 87.62
Li_molar = 6.94

# box dimensions
box_vol = 1 * 1 * 1    # volume (m^3)
P = 0.5                # porosity
rho_s = 1.8            # solid density (g/m^3)
rho_f = 1.0125         # fluid density (g/m^3)
WRR = (rho_f*P)/(rho_s*(1-P))    # water/rock mass ratio
f_frac_box = WRR/(1 + WRR)       # water mass fraction in box
dt = 10                # time step (yr)
t = np.arange(dt, time+1, dt)     # time span 10^6 yr

# reacting solid phase per time step
f_reac_s = pow(math.e, 0-R*(t-dt))-pow(math.e, 0-R*t)
# reacting fluid phase per time step
f_reac = WRR / (WRR + pow(math.e, 0-R*(t-dt))-pow(math.e, 0-R*t))


# Define an initial empty array to facilitate subsequent assignments
# _f_in represents element or isotope composition of fluid in
Ca_f_in = np.ones_like(t)
Mg_f_in = np.ones_like(t)
Sr_f_in = np.ones_like(t)
Li_f_in = np.ones_like(t)
dCa_f_in = np.ones_like(t)
dMg_f_in = np.ones_like(t)
dSr_f_in = np.ones_like(t)
dLi_f_in = np.ones_like(t)

# _dia_b_e represents the composition of newly formed diagenetic carbonates
# _dia_b consists of the remaining primary unstable minerals and stable neomorphism
Ca_dia_b_e = np.ones_like(t)
Ca_dia_b = np.ones_like(t)
Mg_dia_b_e = np.ones_like(t)
Mg_dia_b = np.ones_like(t)
Sr_dia_b_e = np.ones_like(t)
Sr_dia_b = np.ones_like(t)
Li_dia_b_e = np.ones_like(t)
Li_dia_b = np.ones_like(t)
dCa_dia_b_e = np.ones_like(t)
dCa_dia_b = np.ones_like(t)
dMg_dia_b_e = np.ones_like(t)
dMg_dia_b = np.ones_like(t)
dSr_dia_b_e = np.ones_like(t)
dSr_dia_b = np.ones_like(t)
dLi_dia_b_e = np.ones_like(t)
dLi_dia_b = np.ones_like(t)

# D_ represents element distribution coefficient
D_Sr = np.ones_like(t)
D_Li = np.ones_like(t)

# _f_b represents element or isotope composition of fluid out
Ca_f_b = np.ones_like(t)
Mg_f_b = np.ones_like(t)
Sr_f_b = np.ones_like(t)
Li_f_b = np.ones_like(t)
dCa_f_b = np.ones_like(t)
dMg_f_b = np.ones_like(t)
dSr_f_b = np.ones_like(t)
dLi_f_b = np.ones_like(t)

# define a function of diagenetic reaction model over time in the first box
def iteration(Ca_f_i, Mg_f_i, Sr_f_i, Li_f_i, dCa_f_i, dMg_f_i, dSr_f_i, dLi_f_i):

        # diagenetic reaction in time step 1
        # primary fluid (seawater)
        Ca_f_in[0] = Ca_f_i[0]
        Mg_f_in[0] = Mg_f_i[0]
        Sr_f_in[0] = Sr_f_i[0]
        Li_f_in[0] = Li_f_i[0]
        dCa_f_in[0] = dCa_f_i[0]
        dMg_f_in[0] = dMg_f_i[0]
        dSr_f_in[0] = dSr_f_i[0]
        dLi_f_in[0] = dLi_f_i[0]

        # solid phase (Ca, Mg)
        Ca_dia_b_e[0] = (1 - M) * Ca_molar / ((1 - M) * Ca_molar + M * Mg_molar + 60) * 1000000  
        Ca_dia_b[0] = Ca_s_b_0 + (Ca_dia_b_e[0] - Ca_s_b_0) * f_reac_s[0]
        Mg_dia_b_e[0] = M * Mg_molar / ((1 - M) * Ca_molar + M * Mg_molar + 60) * 1000000
        Mg_dia_b[0] = Mg_s_b_0 + (Mg_dia_b_e[0] - Mg_s_b_0) * f_reac_s[0]

        # fluid out (Ca, Mg)
        Ca_f_b[0] = (Ca_s_b_0 * (1 - f_reac[0]) + Ca_f_in[0] * f_reac[0] - (1 - f_reac[0]) * Ca_dia_b_e[0]) / f_reac[0]
        Mg_f_b[0] = (Mg_s_b_0 * (1 - f_reac[0]) + Mg_f_in[0] * f_reac[0] - (1 - f_reac[0]) * Mg_dia_b_e[0]) / f_reac[0]

        # D_ represents the distribution coefficient of single element
        # Kd_Sr represents the distribution coefficient of elements Sr and Ca in equilibrium between solid and fluid phases (set value)
        D_Sr[0] = Ca_dia_b_e[0] * kd_Sr / Ca_f_b[0]    
        D_Li[0] = Ca_dia_b_e[0] * kd_Li / Ca_f_b[0]

        # solid phase (Sr, Li)
        Sr_dia_b_e[0] = (f_reac[0] * Sr_f_in[0] + (1 - f_reac[0]) * Sr_s_b_0) / (f_reac[0] / D_Sr[0] + 1 - f_reac[0])
        Sr_dia_b[0] = Sr_s_b_0 + (Sr_dia_b_e[0] - Sr_s_b_0) * f_reac_s[0]
        Li_dia_b_e[0] = (f_reac[0] * Li_f_in[0] + (1 - f_reac[0]) * Li_s_b_0) / (f_reac[0] / D_Li[0] + 1 - f_reac[0])
        Li_dia_b[0] = Li_s_b_0 + (Li_dia_b_e[0] - Li_s_b_0) * f_reac_s[0]

        # fluid out (Sr, Li)
        Sr_f_b[0] = Sr_dia_b_e[0] / D_Sr[0]
        Li_f_b[0] = Li_dia_b_e[0] / D_Li[0]

        # solid phase (isotope)
        dCa_dia_b_e[0] = ((Ca_f_in[0] * dCa_f_in[0] * f_reac[0] + Ca_s_b_0 * dCa_s_b_0 * (
                1 - f_reac[0])) * alpha_Ca - 1000 * Ca_f_b[0] *
                f_reac[0] * (1 - alpha_Ca)) / (Ca_dia_b_e[0] * (1 - f_reac[0]) * alpha_Ca + Ca_f_b[0] * f_reac[0])
        dCa_dia_b[0] = (dCa_s_b_0 * Ca_s_b_0 + (dCa_dia_b_e[0] * Ca_dia_b_e[0] - dCa_s_b_0 * Ca_s_b_0) * f_reac_s[0]) / \
                Ca_dia_b[0]
        dMg_dia_b_e[0] = ((Mg_f_in[0] * dMg_f_in[0] * f_reac[0] + Mg_s_b_0 * dMg_s_b_0 * (
                1 - f_reac[0])) * alpha_Mg - 1000 * Mg_f_b[0] *
                f_reac[0] * (1 - alpha_Mg)) / (Mg_dia_b_e[0] * (1 - f_reac[0]) * alpha_Mg + Mg_f_b[0] * f_reac[0])
        dMg_dia_b[0] = (dMg_s_b_0 * Mg_s_b_0 + (dMg_dia_b_e[0] * Mg_dia_b_e[0] - dMg_s_b_0 * Mg_s_b_0) * f_reac_s[0]) / \
                Mg_dia_b[0]
        dSr_dia_b_e[0] = ((Sr_f_in[0] * dSr_f_in[0] * f_reac[0] + Sr_s_b_0 * dSr_s_b_0 * (
                1 - f_reac[0])) * alpha_Sr - 1000 * Sr_f_b[0] *
                f_reac[0] * (1 - alpha_Sr)) / (Sr_dia_b_e[0] * (1 - f_reac[0]) * alpha_Sr + Sr_f_b[0] * f_reac[0])
        dSr_dia_b[0] = (dSr_s_b_0 * Sr_s_b_0 + (dSr_dia_b_e[0] * Sr_dia_b_e[0] - dSr_s_b_0 * Sr_s_b_0) * f_reac_s[0]) / \
                Sr_dia_b[0]
        dLi_dia_b_e[0] = ((Li_f_in[0] * dLi_f_in[0] * f_reac[0] + Li_s_b_0 * dLi_s_b_0 * (
                1 - f_reac[0])) * alpha_Li - 1000 * Li_f_b[0] *
                f_reac[0] * (1 - alpha_Li)) / (Li_dia_b_e[0] * (1 - f_reac[0]) * alpha_Li + Li_f_b[0] * f_reac[0])
        dLi_dia_b[0] = (dLi_s_b_0 * Li_s_b_0 + (dLi_dia_b_e[0] * Li_dia_b_e[0] - dLi_s_b_0 * Li_s_b_0) * f_reac_s[0]) / \
                Li_dia_b[0]

        # fluid out (isotope)
        dCa_f_b[0] = (dCa_dia_b_e[0] + 1000) / alpha_Ca - 1000
        dMg_f_b[0] = (dMg_dia_b_e[0] + 1000) / alpha_Mg - 1000
        dSr_f_b[0] = (dSr_dia_b_e[0] + 1000) / alpha_Sr - 1000
        dLi_f_b[0] = (dLi_dia_b_e[0] + 1000) / alpha_Li - 1000

        # iteration remaining time step
        for i in range(1, len(t)):
                
                # fluid composition (fluid in + residual fluid)
                Ca_f_in[i] = Ca_f_i[i] * v * dt + Ca_f_b[i - 1] * (1 - v * dt)          
                Mg_f_in[i] = Mg_f_i[i] * v * dt + Mg_f_b[i - 1] * (1 - v * dt)
                Sr_f_in[i] = Sr_f_i[i] * v * dt + Sr_f_b[i - 1] * (1 - v * dt)
                Li_f_in[i] = Li_f_i[i] * v * dt + Li_f_b[i - 1] * (1 - v * dt)
                dCa_f_in[i] = (Ca_f_i[i] * dCa_f_i[i] * v * dt + Ca_f_b[i - 1] * dCa_f_b[i - 1] * (1 - v * dt)) / Ca_f_in[i]
                dMg_f_in[i] = (Mg_f_i[i] * dMg_f_i[i] * v * dt + Mg_f_b[i - 1] * dMg_f_b[i - 1] * (1 - v * dt)) / Mg_f_in[i]
                dSr_f_in[i] = (Sr_f_i[i] * dSr_f_i[i] * v * dt + Sr_f_b[i - 1] * dSr_f_b[i - 1] * (1 - v * dt)) / Sr_f_in[i]
                dLi_f_in[i] = (Li_f_i[i] * dLi_f_i[i] * v * dt + Li_f_b[i - 1] * dLi_f_b[i - 1] * (1 - v * dt)) / Li_f_in[i]

                # same calculation as time step 1 
                Ca_dia_b_e[i] = (1 - M) * Ca_molar / ((1 - M) * Ca_molar + M * Mg_molar + 60) * 1000000 
                Ca_dia_b[i] = Ca_dia_b[i - 1] + (Ca_dia_b_e[i] - Ca_s_b_0) * f_reac_s[i]
                Mg_dia_b_e[i] = M * Mg_molar / ((1 - M) * Ca_molar + M * Mg_molar + 60) * 1000000
                Mg_dia_b[i] = Mg_dia_b[i - 1] + (Mg_dia_b_e[i] - Mg_s_b_0) * f_reac_s[i]

                Ca_f_b[i] = (Ca_s_b_0 * (1 - f_reac[i]) + Ca_f_in[i] * f_reac[i] - (1 - f_reac[i]) * Ca_dia_b_e[i]) / f_reac[i]
                Mg_f_b[i] = (Mg_s_b_0 * (1 - f_reac[i]) + Mg_f_in[i] * f_reac[i] - (1 - f_reac[i]) * Mg_dia_b_e[i]) / f_reac[i]

                D_Sr[i] = Ca_dia_b_e[i] * kd_Sr / Ca_f_b[i]
                D_Li[i] = Ca_dia_b_e[i] * kd_Li / Ca_f_b[i]

                Sr_dia_b_e[i] = (f_reac[i] * Sr_f_in[i] + (1 - f_reac[i]) * Sr_s_b_0) / (f_reac[i] / D_Sr[i] + 1 - f_reac[i])
                Sr_dia_b[i] = Sr_dia_b[i - 1] + (Sr_dia_b_e[i] - Sr_s_b_0) * f_reac_s[i]
                Li_dia_b_e[i] = (f_reac[i] * Li_f_in[i] + (1 - f_reac[i]) * Li_s_b_0) / (f_reac[i] / D_Li[i] + 1 - f_reac[i])
                Li_dia_b[i] = Li_dia_b[i - 1] + (Li_dia_b_e[i] - Li_s_b_0) * f_reac_s[i]

                Sr_f_b[i] = Sr_dia_b_e[i] / D_Sr[i]
                Li_f_b[i] = Li_dia_b_e[i] / D_Li[i]

                dCa_dia_b_e[i] = ((Ca_f_in[i] * dCa_f_in[i] * f_reac[i] + Ca_s_b_0 * dCa_s_b_0 * (
                        1 - f_reac[i])) * alpha_Ca - 1000 * Ca_f_b[i] * f_reac[i] * (1 - alpha_Ca)) / (
                                        Ca_dia_b_e[i] * (1 - f_reac[i]) * alpha_Ca + Ca_f_b[i] * f_reac[i])
                dCa_dia_b[i] = (dCa_dia_b[i - 1] * Ca_dia_b[i - 1] + (dCa_dia_b_e[i] * Ca_dia_b_e[i] - dCa_s_b_0 * Ca_s_b_0) *
                                f_reac_s[i]) / Ca_dia_b[i]
                dMg_dia_b_e[i] = ((Mg_f_in[i] * dMg_f_in[i] * f_reac[i] + Mg_s_b_0 * dMg_s_b_0 * (
                        1 - f_reac[i])) * alpha_Mg - 1000 * Mg_f_b[i] * f_reac[i] * (1 - alpha_Mg)) / (
                                        Mg_dia_b_e[i] * (1 - f_reac[i]) * alpha_Mg + Mg_f_b[i] * f_reac[i])
                dMg_dia_b[i] = (dMg_dia_b[i - 1] * Mg_dia_b[i - 1] + (dMg_dia_b_e[i] * Mg_dia_b_e[i] - dMg_s_b_0 * Mg_s_b_0) *
                                f_reac_s[i]) / Mg_dia_b[i]
                dSr_dia_b_e[i] = ((Sr_f_in[i] * dSr_f_in[i] * f_reac[i] + Sr_s_b_0 * dSr_s_b_0 * (
                        1 - f_reac[i])) * alpha_Sr - 1000 * Sr_f_b[i] * f_reac[i] * (1 - alpha_Sr)) / (
                                        Sr_dia_b_e[i] * (1 - f_reac[i]) * alpha_Sr + Sr_f_b[i] * f_reac[i])
                dSr_dia_b[i] = (dSr_dia_b[i - 1] * Sr_dia_b[i - 1] + (dSr_dia_b_e[i] * Sr_dia_b_e[i] - dSr_s_b_0 * Sr_s_b_0) *
                                f_reac_s[i]) / Sr_dia_b[i]
                dLi_dia_b_e[i] = ((Li_f_in[i] * dLi_f_in[i] * f_reac[i] + Li_s_b_0 * dLi_s_b_0 * (
                        1 - f_reac[i])) * alpha_Li - 1000 * Li_f_b[i] * f_reac[i] * (1 - alpha_Li)) / (
                                        Li_dia_b_e[i] * (1 - f_reac[i]) * alpha_Li + Li_f_b[i] * f_reac[i])
                dLi_dia_b[i] = (dLi_dia_b[i - 1] * Li_dia_b[i - 1] + (dLi_dia_b_e[i] * Li_dia_b_e[i] - dLi_s_b_0 * Li_s_b_0) *
                                f_reac_s[i]) / Li_dia_b[i]
                dCa_f_b[i] = (dCa_dia_b_e[i] + 1000) / alpha_Ca - 1000
                dMg_f_b[i] = (dMg_dia_b_e[i] + 1000) / alpha_Mg - 1000
                dSr_f_b[i] = (dSr_dia_b_e[i] + 1000) / alpha_Sr - 1000
                dLi_f_b[i] = (dLi_dia_b_e[i] + 1000) / alpha_Li - 1000

        box1 = np.vstack((Ca_f_b, Mg_f_b, Sr_f_b, Li_f_b, dCa_f_b, dMg_f_b, dSr_f_b, dLi_f_b))
        box2 = np.vstack((Ca_dia_b, Mg_dia_b, Sr_dia_b, Li_dia_b, dCa_dia_b, dMg_dia_b, dSr_dia_b, dLi_dia_b))
        return box1, box2

# initial parameter (e.g. Xingmincun formation initial parameter)

Ca_f_int = [764]    
Mg_f_int = [1049.976]

Sr_s_int = [800, 1080]
Sr_f_int = [4, 8]
Li_s_int = [0.28, 0.28]
Li_f_int = [0.053, 0.09]
dLi_s_int = [0, 1.7]
dLi_f_int = [9, 10.7]
kd_Sr_int = [0.05, 0.05]
kd_Li_int = [0.0017, 0.0017]

dCa_f_int = [0]
dMg_f_int = [0.1]
dSr_f_int = [0.708]

Ca_s_int = [389539]
Mg_s_int = [7229]

dCa_s_int = [-1.5]
dMg_s_int = [-0.6]
dSr_s_int = [0.708]

# N represent the number of groups in a formation (e.g. 2 Xingmincun formation was divide into 2 groups)
N = 2
for y in range (N):

        Sr_s_b_0 = Sr_s_int[y]
        Sr_f_i = np.ones_like(t) * Sr_f_int[y]
        Li_s_b_0 = Li_s_int[y]
        Li_f_i = np.ones_like(t) * Li_f_int[y]
        dLi_s_b_0 = dLi_s_int[y]
        dLi_f_i = np.ones_like(t) * dLi_f_int[y]
        kd_Sr = kd_Sr_int[y]
        kd_Li = kd_Li_int[y]

        Ca_s_b_0 = Ca_s_int[0]
        Ca_f_i = np.ones_like(t) * Ca_f_int[0]        
        Mg_s_b_0 = Mg_s_int[0]
        Mg_f_i = np.ones_like(t) * Mg_f_int[0]   
        dCa_s_b_0 = dCa_s_int[0]
        dCa_f_i = np.ones_like(t) * dCa_f_int[0]      
        dMg_s_b_0 = dMg_s_int[0]
        dMg_f_i = np.ones_like(t) * dMg_f_int[0]
        dSr_s_b_0 = dSr_s_int[0]
        dSr_f_i = np.ones_like(t) * dSr_f_int[0]

        box1,box2= iteration(Ca_f_i, Mg_f_i, Sr_f_i, Li_f_i, dCa_f_i, dMg_f_i, dSr_f_i, dLi_f_i)

        # Filter data (reduce flie size)
        datafilter = np.concatenate((np.arange(1, 1000,10) - 1, np.arange(1000, 10000, 100) - 1, np.arange(10000, len(t) + 1, 1000) - 1))
        box_s = np.vstack((t, box2))[:, datafilter] 

        columns = ['t', 'box1_Ca_s', 'box1_Mg_s', 'box1_Sr_s', 'box1_Li_s', 'box1_dCa_s', 'box1_dMg_s','box1_dSr_s','box1_dLi_s']

        ### fluid data
        ## box_f = np.vstack((t, box1))[:, datafilter]
        ## columns_f = ['t', 'box1_Ca_f', 'box1_Mg_f', 'box1_fr_f', 'box1_Li_f', 'box1_dCa_f', 'box1_dMg_f','box1_dSr_f','box1_dLi_f']

        # iterate other boxes
        for j in np.arange(2, boxnum + 1):
                Ca_f_t = box1[0]   
                Mg_f_t = box1[1]
                Sr_f_t = box1[2]
                Li_f_t = box1[3]
                dCa_f_t = box1[4]
                dMg_f_t = box1[5]
                dSr_f_t = box1[6]
                dLi_f_t = box1[7]
 
                box1, box2 = iteration(Ca_f_t, Mg_f_t, Sr_f_t, Li_f_t, dCa_f_t, dMg_f_t, dSr_f_t, dLi_f_t)
                box_s = np.vstack((box_s, box2[:, datafilter])) 
                columns = np.concatenate((columns,['box%d_Ca_s' % (j), 'box%d_Mg_s' % (j), 'box%d_Sr_s' % (j),'box%d_Li_s' % (j),
                                        'box%d_dCa_s' % (j), 'box%d_dMg_s' % (j), 'box%d_dSr_s' % (j),'box%d_dLi_s' % (j)]))
                
                ### fluid data
                ## box_f = np.vstack((box_f, box1[:, datafilter])) 
                ## columns_f = np.concatenate((columns_f,['box%d_Ca_f' % (j), 'box%d_Mg_f' % (j), 'box%d_fr_f' % (j),'box%d_Li_f' % (j),
                ##                             'box%d_dCa_f' % (j), 'box%d_dMg_f' % (j), 'box%d_dSr_f' % (j),'box%d_dLi_f' % (j)]))
                                
        boxdata_s = pd.DataFrame(box_s).T
        boxdata_s.columns = columns
        ## boxdata_f = pd.DataFrame(box_f).T
        ## boxdata_f.columns = columns_f

        colnum = boxdata_s.shape[1]
        rownum = boxdata_s.shape[0]
        
        a = boxdata_s.loc[rownum - 1].values
        Ca = a[np.arange(1, colnum, 8)]
        Mg = a[np.arange(2, colnum, 8)]
        Sr = a[np.arange(3, colnum, 8)]
        Li = a[np.arange(4, colnum, 8)]
        dCa = a[np.arange(5, colnum, 8)]
        dMg = a[np.arange(6, colnum, 8)]
        dSr = a[np.arange(7, colnum, 8)]
        dLi = a[np.arange(8, colnum, 8)]

        diage = pd.DataFrame((Ca, Mg, Sr, Li, dCa, dMg, dSr, dLi)).T
        diage.columns = ['Ca', 'Mg', 'Sr', 'Li', 'dCa', 'dMg', 'dSr', 'dLi']
        
        # save data (modify save path)
        boxdata_s.to_csv(r'C:\Users\boxdata_%d.csv' % (y))
        ## boxdata_f.to_csv(r'C:\Users\boxdata_f_%d.csv' % (y))
        diage.to_csv(r'C:\Users\diagenetic mineral_%d.csv' % (y))

###################################################################################################################
'''Parameters for plot'''

mycolor = ['red','forestgreen','gold','darkorange','dodgerblue']

# Please create a new excel file with the following data in advance
sheet_name_c =['XMC_HC','YCZ_HC','NGL_HC','XMC_MTC','YCZ_MTC','NGL_MTC']

fig, axes = plt.subplots(1,2, figsize=(20,8))
ax1 = axes[0]
ax2 = axes[1]

# plotting sample 
#Formations data
data = pd.read_excel(fr"C:\Users\Formations.xlsx",sheet_name=sheet_name_c[0], header=0)
data_mtc =  pd.read_excel(fr"C:\Users\Formations.xlsx",sheet_name=sheet_name_c[3], header=0)     

for a in range(N):
        ax1.scatter(data['Li/(Ca+Mg)\n(μmol/mol)'], data['d7Li (‰)'], s=30, marker='o', c='forestgreen', ec="black",linewidths=0.5)
        ax2.scatter(data['Sr/(Ca+Mg)\n(mmol/mol)'], data['d7Li (‰)'], s=30, marker='o', c='forestgreen', ec="black",linewidths=0.5)
        ax1.scatter(data_mtc['Li/(Ca+Mg)\n(μmol/mol)'], data_mtc['d7Li (‰)'], s=30, marker='^', c='white',ec="black")
        ax2.scatter(data_mtc['Sr/(Ca+Mg)\n(mmol/mol)'], data_mtc['d7Li (‰)'], s=30, marker='^', c='white',ec="black")

ax1.set_xlabel("Li/(Ca+Mg)(μmol/mol)", fontsize = 'larger')
ax1.set_ylabel('\u03B4$^\mathregular{7}$Li$_\mathregular{}$ (\u2030)', fontsize = 'larger', labelpad = 5)
ax2.set_xlabel("Sr/(Ca+Mg)(mmol/mol)", fontsize = 'larger')
ax2.set_ylabel('\u03B4$^\mathregular{7}$Li$_\mathregular{}$ (\u2030)', fontsize = 'larger', labelpad = 5)

# plotting diagenetic model
for b in np.arange(N):
        data_modelingline = pd.read_csv(r'C:\Users\boxdata_%d.csv'%(b), header=0, index_col=0)
        data_diagemine = pd.read_csv(r'C:\Users\diagenetic mineral_%d.csv'%(b), header=0, index_col=0)

        rownum = data_diagemine.shape[0]

        for c in [1,rownum/10,rownum/5,rownum/2,rownum]:
                ax1.plot(data_modelingline['box%d_Li_s'%(c)]/6.94/(data_modelingline['box%d_Ca_s'%(c)]/40 + data_modelingline['box%d_Mg_s'%(c)] / 24)*1000000, data_modelingline['box%d_dLi_s'%(c)], color='black',linestyle='-',linewidth=0.5)
                ax1.plot(data_diagemine['Li']/6.94/(data_diagemine['Ca']/40+data_diagemine['Mg']/24)*1000000,data_diagemine['dLi'],color=mycolor[b],linestyle='-',linewidth=1)   
                ax2.plot(data_modelingline['box%d_Sr_s'%(c)]/87/(data_modelingline['box%d_Ca_s'%(c)]/40 + data_modelingline['box%d_Mg_s'%(c)] / 24)*1000,data_modelingline['box%d_dLi_s'%(c)],color='black',linestyle='-',linewidth=0.5)
                ax2.plot(data_diagemine['Sr']/87/(data_diagemine['Ca']/40+data_diagemine['Mg']/24)*1000,data_diagemine['dLi'],color=mycolor[b],linestyle='-',linewidth=1)      

plt.show()
plt.savefig(r'C:\Users\fig_ %d.jpg')