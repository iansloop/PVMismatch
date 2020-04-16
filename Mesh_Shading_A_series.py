from pvmismatch import *
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd


def plot_mod_cell_ee(arr, landscape=True):
    from matplotlib import pyplot as plt
    if landscape:
        try:
            M = np.reshape(arr, (8, 12))[::-1]
            M[1::2, :] = M[1::2, ::-1]
        except:
            M = np.reshape(arr, (6, 11))[::-1]
            M[1::2, :] = M[1::2, ::-1]
    else:
        M = np.reshape(arr, (12, 8))[::-1]
        M[1::2, :] = M[1::2, ::-1]

    plt.matshow(M)
    plt.title('bit plot of cell illumination')


def plot_pvm_irr(pvm, landscape=True):
    from matplotlib import pyplot as plt
    P = [p.Ee for p in pvm.pvcells]
    plot_mod_cell_ee(P, landscape=True)


"""Inputs"""
# Represents how much shading the mesh screen provides (0 would let no light through)
mesh_shade = 0.385 #0.427675 #<-- experimental value (did not include denser edges of mesh)

# Represents how much shading a cardboard sheet provides (about 1/8 inch thick)
# (Use 0.01 per Chetan's recommendation for PV Mismatch)
cardboard_shaded = 0.01  # 0.00299 #<-- experimental value

# Represents how much shading two plastic sheets provides (about 1/16 inch thick each)
plastic_shaded = 0.01 #0.001616 #<-- experimental value

# each column contains the indices of the cells within that column. Simplifies code for shading each column
module_columns_X = [tuple(range(0, 12)), tuple(range(12, 24)), tuple(range(24, 36)), tuple(range(36, 48)),
                    tuple(range(48, 60)),
                    tuple(range(60, 72)), tuple(range(72, 84)), tuple(range(84, 96))]
module_columns_A = [tuple(range(0, 11)), tuple(range(11, 22)), tuple(range(22, 33)), tuple(range(33, 44)),
                    tuple(range(44, 55)), tuple(range(55, 66))]

# index corresponds to phase of the shading test. Values are the fraction of the cell covered by the mesh
shaded_area_X = np.array([0, 0.5, 1, 1, 1, 1, 1, 1, 1, 1])
shaded_area_A = np.array([0, 0.3951, 0.7902, 0.5805, 0.3708, 0.1611, 0.9513, 0.7416, 0.5318, 1])

# average irradiance before test starts
inital_irr = 0.71436

# average irradiance of west and east reference cell on roof (in suns, where 1 sun = 1000 W/m^2)
natural_irrad = np.array([inital_irr, 0.74796, 0.703228035, 0.752545, 0.75282, 0.751505, 0.746345, 0.742944995, 0.71332001, 0.7106])

# input experimental data for Pmp, Vmp, Imp
MPPT_X_exp = pd.read_excel(r'C:\Users\isloop\OneDrive for Business\Desktop\PVMismatch_Resources\NGT_XSERIES_POWER.xlsx')
MPPT_A_exp = pd.read_excel(r'C:\Users\isloop\OneDrive for Business\Desktop\PVMismatch_Resources\A_SERIES_STRING_POWER.xlsx')

# pull dc monitoring data from the east so that the power from the west can be normalized to it
East_DC_Power = pd.read_excel(
    r'C:\Users\isloop\OneDrive for Business\Desktop\PVMismatch_Resources\XSERIES_EAST_DC_MOD_POWER.xlsx')

# create series to serve as horizontal axis for 10 min averaged data
ten_min_interval = MPPT_X_exp['Timestamps'].iloc[20:120:10]
tmi = [ten_min_interval.iloc[j].tz_localize('UTC') for j in range(0,len(ten_min_interval))]
tmi = [tmi[k].tz_convert('America/Los_Angeles') for k in range(0,len(tmi))]
time_axis = [tmi[l].strftime('%H:%M') for l in range(0, len(tmi))]

# amount of irradiance received by the most recently shaded column
irrad_pattern_X = natural_irrad * (1 - shaded_area_X + shaded_area_X * mesh_shade)
irrad_pattern_A = natural_irrad * (1 - shaded_area_A + shaded_area_A * mesh_shade)

# average operating temperature of the cells during each phase
temp_array_X = np.array([31.679, 32.257, 33.582, 34.156, 34.569, 35.503, 36.933, 37.587, 37.546]) + np.array(
    [273.15, ] * 9)  # Kelvin
temp_array_A = np.array([31.947, 33.865, 35.373, 35.599, 36.262, 36.83, 37.882, 38.772, 37.897]) + np.array(
    [273.15, ] * 9)  # Kelvin

# temperature of the system before test starts
nominal_temp_X = 30.285 + 273.15  # kelvin
nominal_temp_A = 29.911 + 273.15  # kelvin

# irradiance received by a fully mesh-shaded cell
full_mesh_shade = mesh_shade * natural_irrad


    #A series
#A series (NGT) cell and module properties
# Adam's values
RS = 0.0046  # [ohm] series resistance
RSH = 100  # [ohm] shunt resistance
ISAT1_T0 = 4.35e-12  # [A] diode one saturation current
ISAT2_T0 = 1.85e-07  # [A] diode two saturation current
ISC0_T0 = 10.96  # [A] reference short circuit current
#TCELL = 298.15  # [K] cell temperature
ARBD = 1.036748445065697E-4  # reverse breakdown coefficient 1
BRBD = 0.  # reverse breakdown coefficient 2
VRBD_ = -5.527260068445654  # [V] reverse breakdown voltage
NRBD = 3.284628553041425  # reverse breakdown exponent
EG = 1.1  # [eV] band gap of cSi
ALPHA_ISC = 0.0003551  # [1/K] short circuit current temperature coefficient
#NPTS = 1500
cellArea = 258.26
# Module parameters
#NUMBERCELLS = 66
Vbypass = np.float64(-0.5)  # [V] trigger voltage of bypass diode
#MODULEAREA = 1.8629  # [m2]
# Tamir's email values
'''
ISC0_T0=10.2
RS=6.2e-3
RSH=38.7
ISAT1_T0=6.46e-12
ISAT2_T0=2.58e-7
ARBD=2.04
BRBD=-0.6588
VRBD=-11.8211
NRBD=10
cellArea=258.25
EG = 1.166
ALPHA_ISC = 0.0003551
Vbypass = -0.5
'''
# David Jacob's values
'''
ISC0_T0=10.2
RS=6.2e-3
RSH=38.7
ISAT1_T0=6.46e-12
ISAT2_T0=2.58e-7
ARBD=0.63
BRBD=0.52
VRBD=-6.00
NRBD=1.22
cellArea=258.3
EG = 1.166
ALPHA_ISC = 0.0003551
Vbypass = -0.5
'''
# Tamir's table values
'''
ISC0_T0=10.0
RS=0.0032
RSH=8.025
ISAT1_T0=4.151E-12
ISAT2_T0=2.702E-06
ARBD=1.735E-02
BRBD=-0.6588
VRBD=-4.50
NRBD=3.926
cellArea=258.25
EG = 1.166
ALPHA_ISC = 0.0003551
Vbypass = -0.5
'''

#create non-standard NGT cell and PV system with inital irradiance and temperature
pvcell_A = pvcell.PVcell(Rs=RS, Rsh=RSH, Isat1_T0=ISAT1_T0, Isat2_T0=ISAT2_T0,
                 Isc0_T0=ISC0_T0, aRBD=ARBD, VRBD=VRBD,
                 nRBD=NRBD, bRBD=BRBD, Eg=EG, alpha_Isc=ALPHA_ISC)
pvmodule_A = pvmodule.PVmodule(cell_pos=pvmodule.standard_cellpos_pat(11,[6]), Vbypass=[Vbypass], pvcells=pvcell_A, cellArea=cellArea)
pvsys_A = pvsystem.PVsystem(numberStrs=1,numberMods=4,pvmods=pvmodule_A)
pvsys_A_control = pvsystem.PVsystem(numberStrs=1,numberMods=4,pvmods=pvmodule_A)
plt.ion()
pvsys_A.setSuns(inital_irr)
pvsys_A_control.setSuns(inital_irr)
pvsys_A.setTemps(nominal_temp_A)
pvsys_A_control.setTemps(nominal_temp_A)

# shade the two modules at the ends of pvsys_A to simulate the plastic shading
#per Chetan's recommendation, don't set suns lower than 0.01
pvsys_A.setSuns({0: {0:inital_irr*plastic_shaded, 3:inital_irr*plastic_shaded}})

# record initial Pmp, Imp, Vmp (AC-String level). First String is shaded, second string is unshaded control
MPPT_A = pd.DataFrame(
    columns=['Pmp', 'Pmp_control', 'Pmp_norm', 'Vmp', 'Vmp_control', 'Vmp_norm', 'Imp', 'Imp_control', 'Imp_norm'],
    index=range(0, 10))
MPPT_A.iloc[0] = {'Pmp': sum([pvsys_A.pvmods[0][j].Pmod.max() for j in range(0,4)]), 'Vmp': pvsys_A.Vmp,
                  'Imp': pvsys_A.Imp,
                  'Pmp_control': sum([pvsys_A_control.pvmods[0][j].Pmod.max() for j in range(0,4)]), 'Vmp_control': pvsys_A_control.Vmp,
                  'Imp_control': pvsys_A_control.Imp}

#shade cells, update temperatures, and record power for each phase of the test
for i in range(0,9):
    #pvsys_A.pvmods[0][1].plotCell()
    #pvsys_A.pvmods[0][1].plotMod()
    # update irradiance on all cells in string 1, module 2 and 3 (mesh shaded modules)
    pvsys_A.setSuns(
        {0: {1: {'cells': tuple(range(0, 66)),
                 'Ee': tuple([pvsys_A.pvmods[0][1].Ee[j][0] * natural_irrad[i + 1] / natural_irrad[i] for j in
                              range(0, 66)])},
             2: {'cells': tuple(range(0, 66)),
                 'Ee': tuple([pvsys_A.pvmods[0][2].Ee[j][0] * natural_irrad[i + 1] / natural_irrad[i] for j in
                              range(0, 66)])}}})

    #update irradiance on string 1, modules 1 and 4 (plastic shaded modules), and string 2 (unshaded string)
    pvsys_A.setSuns({0: {0: natural_irrad[i+1]*plastic_shaded, 3: natural_irrad[i+1]*plastic_shaded}})
    pvsys_A_control.setSuns(natural_irrad[i+1])

    if i == 0 or i == 1:
        pvsys_A.setSuns({0: {1: [(irrad_pattern_A[i+1],)*11,module_columns_A[-1]],
                         2: [(irrad_pattern_A[i+1],)*11,module_columns_A[-1]]}})
    elif i == 4 or i == 5:
        pvsys_A.setSuns({0: {1: [(full_mesh_shade[i+1],) * 11, module_columns_A[-3]],
                         2: [(full_mesh_shade[i+1],) * 11, module_columns_A[-3]]}})
        pvsys_A.setSuns({0: {1: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-4]],
                         2: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-4]]}})
    elif i == 6 or i == 7:
        pvsys_A.setSuns({0: {1: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i+2]],
                         2: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i+2]]}})
        pvsys_A.setSuns({0: {1: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-i+1]],
                         2: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-i+1]]}})
    elif i == 8:
        pvsys_A.setSuns({0: {1: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i + 2]],
                         2: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i + 2]]}})
    else:
        pvsys_A.setSuns({0: {1: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i+1]],
                         2: [(full_mesh_shade[i+1],) * 11, module_columns_A[-i+1]]}})
        pvsys_A.setSuns({0: {1: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-i]],
                         2: [(irrad_pattern_A[i+1],) * 11, module_columns_A[-i]]}})
    pvsys_A.setTemps(temp_array_A[i])
    pvsys_A_control.setTemps(temp_array_A[i])

    MPPT_A.iloc[i + 1] = {'Pmp': sum([pvsys_A.pvmods[0][j].Pmod.max() for j in range(0,4)]), 'Vmp': pvsys_A.Vmp,
                  'Imp': pvsys_A.Imp,
                  'Pmp_control': sum([pvsys_A_control.pvmods[0][j].Pmod.max() for j in range(0,4)]), 'Vmp_control': pvsys_A_control.Vmp,
                  'Imp_control': pvsys_A_control.Imp}

    MPPT_A['Pmp_norm'] = MPPT_A['Pmp'] / MPPT_A['Pmp_control']
    MPPT_A['Vmp_norm'] = MPPT_A['Vmp'] / MPPT_A['Vmp_control']
    MPPT_A['Imp_norm'] = MPPT_A['Imp'] / MPPT_A['Imp_control']

    MPPT_A_exp['A_series_norm'] = MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_4'] / MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_3']
'''
# plotting
'''
# MPPT_A_experimental = np.array([506.465182,400.882196,332.6838467,304.707138,232.7229941,217.095214,209.072656,202.431444,188.462066,182.377072])
# error_A = (MPPT_A-MPPT_A_experimental)/MPPT_A_experimental*100
plt.figure()
#plt.subplot(211)
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_A['Pmp_norm'], 'bo', label='PV Mismatch')
plt.plot(MPPT_A_exp['Timestamps'].iloc[14:107], MPPT_A_exp['A_series_norm'].iloc[14:107], label='A Series (NGT)')
# plt.plot(error_X,label='Error (%)')
plt.legend()
plt.title('Performance of 4-Module NGT AC String vs Width of Column Mesh Shading')
#plt.xlabel('Shading Width (in) and Start/Stop Timestamps (XX:XX)')
plt.xticks(ten_min_interval - pd.Timedelta(minutes=5), [])
plt.annotate(s='2 Modules Shaded by Plastic \n Covers, no Mesh Shading',
             xy=(ten_min_interval.iloc[0]-pd.Timedelta(minutes=5), 0.5),
             xytext=(ten_min_interval.iloc[0], 0.6),
             arrowprops={'arrowstyle': '->'})
plt.ylabel('Normalized AC String Power')
plt.yticks(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
plt.grid(True)

#plot error
reduced_MPPT_A_exp = MPPT_A_exp.iloc[15:115:10]
MPPT_A['A_series_error'] = np.array([(MPPT_A['Pmp_norm'].iloc[i] - reduced_MPPT_A_exp['A_series_norm'].iloc[i])/reduced_MPPT_A_exp['A_series_norm'].iloc[i] for i in range(0,10)])*100

# plt.subplot(212)
# plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_A['A_series_error'], '*',label='A Series (NGT)')
# plt.legend()
# plt.title('Error Between PV Mismatch and A Series Experimental Values')
# plt.xlabel('Shading Width (in)')
# plt.xticks(ten_min_interval.loc[20:110] - pd.Timedelta(minutes=5), ['0','2.5','5','10','15','20','25','30','35','40'])
# plt.ylabel('% Error')
# #plt.yticks([y/10.0 for y in range(0,110,10)], [y/10.0 for y in range(0,110,10)])
# plt.grid(True)
# plt.xticks(ten_min_interval.loc[20:110] - pd.Timedelta(minutes=5), ['0','2.5','5','10','15','20','25','30','35','40'])
# plt.ylabel('% Error')
# plt.yticks([-2,-1,0,1,2,3,4,5,6])
# #plt.yticks([y/10.0 for y in range(0,110,10)], [y/10.0 for y in range(0,110,10)])
# plt.grid(True)

table = plt.table([np.round(MPPT_A['A_series_error'],3)],
          colLabels=['0in','2.5in','5in','10in','15in','20in','25in','30in','35in','40in'],
          rowLabels=['A Series Error (%)'], fontsize=12)

plt.subplots_adjust(left=0.2, bottom=0.2)

#extra plots

plt.figure()
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_A['Pmp'], 'bo', label='PV Mismatch Pmp')
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_A['Pmp_control'], 'b*', label='PV Mismatch Pmp_control')
plt.plot(MPPT_A_exp['Timestamps'], MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_4']*1000, label='Pmp')
plt.plot(MPPT_A_exp['Timestamps'], MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_3']*1000, label='Pmp_control')
plt.legend()

plt.xlabel('Test Interval')
plt.xticks(ten_min_interval, time_axis)
plt.ylabel('AC String Power')
plt.grid(True)

sample_A = MPPT_A_exp['SPDA.RND.ACM_ROOF.Power_4'].iloc[15:115:10]*1000
error_A = ((np.array([j for j in sample_A]) - np.array([i for i in MPPT_A['Pmp']]))/np.array([j for j in sample_A]))*100

# pvsys_A.pvmods[0][3].plotMod()
# pvsys_A.pvmods[0][0].plotMod()
# pvsys_A.plotSys()

# IN PROGRESS
# experimental_power = np.array([0.4524,0.352,0.2649,0.2436,0.2234,0.2225,0.2163,0.2004,0.1925])*1000
# error = (MPPT_A-experimental_power)/experimental_power*100
