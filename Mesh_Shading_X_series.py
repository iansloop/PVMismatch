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

""""Simulation"""
# X series (Gen E)
# X series cell and module properties
#'''
RS=0.008
RSH=250.01226369025448
ISAT1_T0=2.974132024e-12
ISAT2_T0=2.394153128e-7
ISC0_T0=6.3056#6.590375
ARBD=1.036748445065697e-4
BRBD=0#-0.6588
VRBD=-5.527260068445654
NRBD=3.284628553041425
ALPHA_ISC=0.0003551
CELLAREA=153.33
EG=1.1
Vbypass = -0.5
#'''
'''
RS=0.0022904554199000655
RSH=5.524413919705285
ISAT1_T0=2.6951679883577537e-12
ISAT2_T0=9.078875806333005e-7
ISC0_T0=6.590375
ARBD=1.036748445065697e-4
BRBD=-0.6588
VRBD=-5.527260068445654
NRBD=3.284628553041425
ALPHA_ISC=0.0003551
CELLAREA=153.33
EG=1.1
Vbypass = -0.5
'''
'''
RS=0.00477#0.0022904554199000655
RSH=14.866#5.524413919705285
ISAT1_T0=5.615E-12#2.6951679883577537e-12
ISAT2_T0=6.133E-7#9.078875806333005e-7
ISC0_T0=6.39#6.590375
ARBD=1.735E-2
BRBD=-0.6588
VRBD=-4.50
NRBD=3.926
ALPHA_ISC=0.0003551
CELLAREA=153.33
EG=1.166
Vbypass = -0.5
'''
'''
# Datasheet
RS = 0.00477  # 0.0022904554199000655
RSH = 14.866  # 5.524413919705285
ISAT1_T0 = 5.615E-12  # 2.6951679883577537e-12
ISAT2_T0 = 6.133E-7  # 9.078875806333005e-7
ISC0_T0 = 6.66  # 6.590375
ARBD = 1.735E-2
BRBD = -0.6588
VRBD = -4.50
NRBD = 3.926
ALPHA_ISC = 0.0003551
CELLAREA = 153.33
EG = 1.166
Vbypass = -0.5
'''
VRBD = np.float64(VRBD)
# Create PV system with inital irradiance and temperature
pvcell_X = pvcell.PVcell(Rs=RS, Rsh=RSH, Isat1_T0=ISAT1_T0, Isat2_T0=ISAT2_T0,
                         Isc0_T0=ISC0_T0, aRBD=ARBD, bRBD=BRBD, VRBD=VRBD,
                         nRBD=NRBD, Eg=EG, alpha_Isc=ALPHA_ISC)
pvmodule_X = pvmodule.PVmodule(pvcells=pvcell_X, Vbypass=[Vbypass, Vbypass, Vbypass], cellArea=CELLAREA)
pvsys_X = pvsystem.PVsystem(numberStrs=1, numberMods=1, pvmods=pvmodule_X)
pvsys_X_control = pvsystem.PVsystem(numberStrs=1, numberMods=1, pvmods=pvmodule_X)
# pvsys_X = pvsystem.PVsystem(numberStrs=2,numberMods=4)
plt.ion()
pvsys_X.setSuns(inital_irr)
pvsys_X.setTemps(nominal_temp_X)
pvsys_X_control.setSuns(inital_irr)
pvsys_X_control.setTemps(nominal_temp_X)

# shade first (index=0) and last (index=3) module completely on both X series strings (moving left to right)
# pvsys_X.setSuns({0: {0:cardboard_shaded*inital_irr, 3:cardboard_shaded*inital_irr}, 1:{0:cardboard_shaded*inital_irr, 3:cardboard_shaded*inital_irr}})
# per Chetan's recommendation, don't set suns lower than 0.01
#pvsys_X.setSuns({0: {0: 0.01, 3: 0.01}, 1: {0: 0.01, 3: 0.01}})

# record initial Pmp, Imp, Vmp. First String is shaded, second string is unshaded control
MPPT_X = pd.DataFrame(
    columns=['Pmp', 'Pmp_control', 'Pmp_norm_Type_D', 'Vmp', 'Vmp_control', 'Vmp_norm', 'Imp', 'Imp_control', 'Imp_norm'],
    index=range(0, 10))
MPPT_X.iloc[0] = {'Pmp': pvsys_X.Pmp, 'Vmp': pvsys_X.Vmp,
                  'Imp': pvsys_X.Imp,
                  'Pmp_control': pvsys_X_control.Pmp, 'Vmp_control': pvsys_X_control.Vmp,
                  'Imp_control': pvsys_X_control.Imp}

# shade cells, update temperatures, and record power for each phase of the test
for i in range(0, 9):
    if i == 2:
        pvmodule_X_5in = pvsys_X.pvmods[0][0]
    if i == 3:
        pvmodule_X_10in = pvsys_X.pvmods[0][0]
    #pvsys_X.pvmods[0][0].plotCell()
    #pvsys_X.pvmods[0][0].plotMod()
    # update the natural irradiance across ALL cells of 1st string, 2nd module
    pvsys_X.setSuns(
        {0: {0: {'cells': tuple(range(0, 96)),
                 'Ee': tuple([pvsys_X.pvmods[0][0].Ee[j][0] * natural_irrad[i+1] / natural_irrad[i] for j in
                              range(0, 96)])}}})

    #update natural irradiance on the unshaded control module
    pvsys_X_control.setSuns({0: {0: natural_irrad[i+1]}})
    #update cell temperatures
    pvsys_X_control.setTemps(temp_array_X[i])

    if i == 0:
        pvsys_X.setSuns({0: {0: {'cells': module_columns_X[0], 'Ee': (irrad_pattern_X[i+1],) * 12}}})
    else:
        # shading the rest of the columns
        pvsys_X.setSuns({0: {0: {'cells' : module_columns_X[i - 1], 'Ee' : (irrad_pattern_X[i+1],) * 12}}})

    #plot_pvm_irr(pvsys_X.pvmods[0][1])
    # shading half of first column

    MPPT_X.iloc[i+1] = {'Pmp': pvsys_X.Pmp, 'Vmp': pvsys_X.Vmp,
                      'Imp': pvsys_X.Imp,
                      'Pmp_control': pvsys_X_control.Pmp, 'Vmp_control': pvsys_X_control.Vmp,
                      'Imp_control': pvsys_X_control.Imp}

MPPT_X['Pmp_norm_Type_D'] = MPPT_X['Pmp'] / MPPT_X['Pmp_control']
MPPT_X['Vmp_norm'] = MPPT_X['Vmp'] / MPPT_X['Vmp_control']
MPPT_X['Imp_norm'] = MPPT_X['Imp'] / MPPT_X['Imp_control']

MPPT_X_exp['Type_D_norm'] = MPPT_X_exp['W5 Type D'] / East_DC_Power['Power E5 Type D']
MPPT_X_exp['Type_E_norm'] = MPPT_X_exp['W6 Type E'] / East_DC_Power['Power E6 Type E']

# manually enter the pmp for type E where it didnt find the global MPPT (approximate)
MPPT_X['Pmp_norm_Type_E'] = MPPT_X['Pmp_norm_Type_D']
#MPPT_X['Pmp_norm_Type_E'][2] = 107.99/MPPT_X['Pmp_control'][2]
#MPPT_X['Pmp_norm_Type_E'][3] = 112.25/MPPT_X['Pmp_control'][3]

#MPPT_X['Exp_average'] =

# plotting
# error_X = (MPPT_X['Pmp']-MPPT_X_experimental)/MPPT_X_experimental*100

# plot Pmp
custom_interval = np.concatenate((pd.Series(ten_min_interval.iloc[0]-pd.Timedelta(minutes=5)), pd.Series(ten_min_interval.iloc[0]), ten_min_interval.loc[30:110] - pd.Timedelta(minutes=5), pd.Series(ten_min_interval.iloc[-1])))
custom_axis = np.concatenate((['0'], [time_axis[0]], ['2.5','5','10','15','20','25','30','35','40'], [time_axis[-1]]))

fig, ax = plt.subplots()
#plt.subplot(211)
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Pmp_norm_Type_D'], 'b*', label='PV Mismatch')
#plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Pmp_norm_Type_E'], 'ro', label='PV Mismatch Type E')
plt.plot(MPPT_X_exp['Timestamps'].iloc[14:107], MPPT_X_exp['Type_D_norm'].iloc[14:107], 'b--', label='Type D')
plt.plot(MPPT_X_exp['Timestamps'].iloc[14:107], MPPT_X_exp['Type_E_norm'].iloc[14:107],'r-', label='Type E')
# plt.plot(error_X,label='Error (%)')
plt.legend()
plt.title('X-Series Type E and D Module Performance vs Width of Column Shading')
#plt.xlabel('Shading Width (in) and Start/Stop Timestamps (XX:XX)')
plt.xticks(ten_min_interval - pd.Timedelta(minutes=5), [])
plt.ylabel('Normalized DC Power')
plt.yticks(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
plt.grid(True)

#plot error
reduced_MPPT_X_exp = MPPT_X_exp.iloc[15:115:10]
MPPT_X['Type_D_error'] = np.array([(MPPT_X['Pmp_norm_Type_D'].iloc[i]
                                    - reduced_MPPT_X_exp['Type_D_norm'].iloc[i])/reduced_MPPT_X_exp['Type_D_norm'].iloc[i] for i in range(0,10)])*100
MPPT_X['Type_E_error'] = np.array([(MPPT_X['Pmp_norm_Type_E'].iloc[i]
                                    - reduced_MPPT_X_exp['Type_E_norm'].iloc[i])/reduced_MPPT_X_exp['Type_E_norm'].iloc[i] for i in range(0,10)])*100

#plt.subplot(212)
table = plt.table([np.round(MPPT_X['Type_D_error'],3), np.round(MPPT_X['Type_E_error'],3)],
          colLabels=['0in','2.5in','5in','10in','15in','20in','25in','30in','35in','40in'],
          rowLabels=['Type D Error (%)', 'Type E Error (%)'], fontsize=12)

plt.subplots_adjust(left=0.2, bottom=0.2)
# plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Type_D_error'], 'b*',label='Type D')
# plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Type_E_error'], 'ro',label='Type E')
# plt.legend()
# plt.title('Error Between PV Mismatch and Type D and E Experimental Values')
# plt.xlabel('Shading Width (in)')
# plt.xticks(ten_min_interval.loc[20:110] - pd.Timedelta(minutes=5), ['0','2.5','5','10','15','20','25','30','35','40'])
# plt.ylabel('% Error')
#plt.yticks([y/10.0 for y in range(0,110,10)], [y/10.0 for y in range(0,110,10)])
#plt.grid(True)

#
# # plot Imp
plt.figure()
plt.subplot(211)
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Imp'], 'b*', label='PV Mismatch')
plt.plot(MPPT_X_exp['Timestamps'],
         MPPT_X_exp['SPDA.RND.DCM_ROOF.StrCurrent_W5'],
         label='Experimental')
# plt.plot(error_X,label='Error (%)')
plt.legend()
plt.xlabel('Test Interval')
plt.xticks(ten_min_interval)
plt.ylabel('DC Imp (A)')
plt.grid(True)

# plot Vmp
plt.subplot(212)
plt.plot(ten_min_interval - pd.Timedelta(minutes=5), MPPT_X['Vmp'], 'b*', label='PV Mismatch')
plt.plot(MPPT_X_exp['Timestamps'], MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W5'], label='Experimental')
# plt.plot(error_X,label='Error (%)')
plt.legend()
plt.xlabel('Test Interval')
plt.xticks(ten_min_interval)
plt.ylabel('DC Vmp (V)')
plt.grid(True)

#  plot discrepant IV/PV curves at 5 and 10 in shade
# 5 in
fig, axs = plt.subplots(2, 1)
fig.suptitle('5in Column Mesh Shading X-Series', fontsize=20)
axs[0].plot(pvmodule_X_5in.Vmod, pvmodule_X_5in.Imod, label='PV Mismatch')
axs[0].plot(reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W5'].iloc[2], reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrCurrent_W5'].iloc[2], 'b*', label='Type D Experimental')
axs[0].plot(reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W6'].iloc[2], reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrCurrent_W6'].iloc[2], 'ro', label='Type E Experimental')
axs[0].set_title('Module I-V Characteristics')
axs[0].set_ylabel('Module Current, I [A]')
axs[0].set_ylim(ymin=0)
axs[0].set_xlim(pvmodule_X_5in.Vmod.min() - 1, pvmodule_X_5in.Vmod.max() + 1)
axs[0].legend()
axs[0].grid()

axs[1].plot(pvmodule_X_5in.Vmod, pvmodule_X_5in.Pmod, label='PV Mismatch')
axs[1].plot(reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W5'].iloc[2], reduced_MPPT_X_exp['W5 Type D'].iloc[2], 'b*', label='Type D Experimental')
axs[1].plot(reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W6'].iloc[2], reduced_MPPT_X_exp['W6 Type E'].iloc[2], 'ro', label='Type E Experimental')
axs[1].set_title('Module P-V Characteristics')
axs[1].set_xlabel('Module Voltage, V [V]')
axs[1].set_ylabel('Module Power, P [W]')
axs[1].set_ylim(ymin=0)
axs[1].set_xlim(pvmodule_X_5in.Vmod.min() - 1, pvmodule_X_5in.Vmod.max() + 1)
axs[1].legend()
axs[1].grid()

# 10 in
fig, axs = plt.subplots(2, 1)
fig.suptitle('10in Column Mesh Shading X-Series', fontsize=20)
axs[0].plot(pvmodule_X_10in.Vmod, pvmodule_X_10in.Imod, label='PV Mismatch')
axs[0].plot(reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W5'].iloc[3], reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrCurrent_W5'].iloc[3], 'b*', label='Type D Experimental')
axs[0].plot(reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W6'].iloc[3], reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrCurrent_W6'].iloc[3], 'ro', label='Type E Experimental')
axs[0].set_title('Module I-V Characteristics')
axs[0].set_ylabel('Module Current, I [A]')
axs[0].set_ylim(ymin=0)
axs[0].set_xlim(pvmodule_X_10in.Vmod.min() - 1, pvmodule_X_10in.Vmod.max() + 1)
axs[0].legend()
axs[0].grid()

axs[1].plot(pvmodule_X_10in.Vmod, pvmodule_X_10in.Pmod, label='PV Mismatch')
axs[1].plot(reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W5'].iloc[3], reduced_MPPT_X_exp['W5 Type D'].iloc[3], 'b*', label='Type D Experimental')
axs[1].plot(reduced_MPPT_X_exp['SPDA.RND.DCM_ROOF.StrVoltage_W6'].iloc[3], reduced_MPPT_X_exp['W6 Type E'].iloc[3], 'ro', label='Type E Experimental')
axs[1].set_title('Module P-V Characteristics')
axs[1].set_xlabel('Module Voltage, V [V]')
axs[1].set_ylabel('Module Power, P [W]')
axs[1].set_ylim(ymin=0)
axs[1].set_xlim(pvmodule_X_10in.Vmod.min() - 1, pvmodule_X_10in.Vmod.max() + 1)
axs[1].legend()
axs[1].grid()

# pvsys_X.pvmods[0][1].plotMod()
# pvsys_X.plotSys()