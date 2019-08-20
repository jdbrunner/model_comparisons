# %run load_gore_results
# #Creates a DataFrame of the observed outcomes of the trio experiments (real_outs)
# #Creates a DataFrame of the observed outcomes of the pair experiment "by the eye test" (pair_outs_gore)
# #Creates a DataFrame of the reported k&r values from the Gore paper (mono_params_gore)
# #Creates a DataFrame of the reported parameters from the Gore paper (interacts_gore_unscld)
# #Creates a DataFrame of the reported parameters after model rescaling (interacts_gore)
# #Creates a DataFrame of the pair model analysis (pair_out_gore)
# #Creates a DataFrame of the reported and rescaled parameters adjusted to match pair observations (interacts_gore_pf)
# #Creates a DataFrame of the single species timecourse experiments (mono_data)
# #Creates a DataFrame of the pair experiment timecourse data (pair_data)
# #Creates a DataFrame of the trio experiment (only start/end) (trio_data)
# #Creates a DataFrame of the group experiment data (only start/end) (groups_data)

import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.integrate import ode
from scipy.interpolate import interp1d
from numpy.random import rand
from lv_pair_trio_functs import *
from load_gore_results import *
from contextlib import redirect_stdout

import seaborn as sb

mono_params_df = pd.read_csv('mono_parameters.csv',index_col  = 0)
pair_outs_fit = pd.read_csv('pair_outcomes.csv',index_col  = 0)
interact_params = pd.read_csv('lotka_volterra_fitted.csv',index_col  = 0)
interact_params_search = pd.read_csv('lotka_volterra_search.csv',index_col  = 0)

exp_index = list(real_outs.index)


spa = 'Ea'
spb = 'Pa'
spc = 'Sm'

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

fig, axs = plt.subplots(1,1, figsize = (8,8))
plt.rcParams.update({'font.size': 15})


for ii in pair_data.keys():
    spes = ii.split('_')
    for exper in pair_data[ii].index.levels[0]:
        for il in range(len(pair_data[ii].loc[exper])-1):
            if pair_data[ii].loc[exper].iloc[il,0] ==0:
                pair_data[ii].loc[exper].iloc[il,0] = np.mean(pair_data[ii].loc[exper].iloc[il:,0])
            if pair_data[ii].loc[exper].iloc[il,1] ==0:
                pair_data[ii].loc[exper].iloc[il,1] = np.mean(pair_data[ii].loc[exper].iloc[il:,1])

pair_data_scld = {}

for ii in pair_data.keys():
    pair_data_scld[ii] = pair_data[ii].copy()
    spes = ii.split('_')
    pair_data_scld[ii].loc[:,spes[0]] = pair_data_scld[ii].loc[:,spes[0]]/mono_params_df.loc[spes[0],'K']
    pair_data_scld[ii].loc[:,spes[1]] = pair_data_scld[ii].loc[:,spes[1]]/mono_params_df.loc[spes[1],'K']



def avg_growth_mono_data(m_data):
    a_growths = np.array([avg_xdoverx(m_data[Col].values) for Col in m_data.columns])
    return a_growths


def avg_xdoverx(experim,tstp = 1):
    '''experim is an array of values representing a time course of biomass. Calculates the average growth rate per
    biomass'''
    #first filter the NaN
    experim = experim[np.invert(np.isnan(experim))]
    grates = np.array([experim[i+1] - experim[i] for i in range(len(experim)-1)])/tstp
    #avoid dividing by 0
    nozs = np.where(experim == 0)
    experim[nozs] = 1
    grates_per_org = np.array(grates)/experim[:-1]
    return grates_per_org


avg_growth_mono = avg_growth_mono_data(mono_data[spa])
monomean = np.mean(avg_growth_mono, axis = 0)
monostd = np.std(avg_growth_mono, axis = 0)



x = np.arange(1,avg_growth_mono.shape[1]+1)
pltrng = np.arange(1,avg_growth_mono.shape[1],0.1)

y1 = interp1d(x,monomean, kind = 'cubic')
y2 = interp1d(x,monostd,kind = 'cubic')
#
# axs.plot(pltrng,y1(pltrng))
# axs.fill_between(pltrng,y1(pltrng)+y2(pltrng),y1(pltrng)-y2(pltrng),alpha = 0.3)



kyli = np.array(list(trio_data.keys()))

kyhere = kyli[np.where([spa in ky and spb in ky and spc in ky for ky in kyli])][0]
exp_df = trio_data[kyhere]



avg_growth_trio = (exp_df.xs(5.0,level = 1) - exp_df.xs(0.0,level = 1))/(5*exp_df.xs(0.0,level = 1))
avg_growth_trio = avg_growth_trio.loc[:,spa].values

trio_mean = np.array([np.mean(avg_growth_trio)]*len(pltrng)) -y1(pltrng)
trio_std = np.array([np.std(avg_growth_trio)]*len(pltrng)) - y2(pltrng)



axs.plot(pltrng,trio_mean, color = 'c', label = 'In Trio', lw = 3)
axs.fill_between(pltrng, trio_mean + trio_std, trio_mean - trio_std, alpha = 0.3, color = 'c', label = '_in trio')



kylipair = np.array(list(pair_data.keys()))
#
#
#
def avg_growth_pair_data(p_data):
    exp_nums = p_data.index.levels[0]
    a_growths = [list(avg_xdoverx(p_data.loc[l][spa].values)) for l in exp_nums]
    for rwi in range(len(a_growths)):
        rw = a_growths[rwi]
        if len(rw) < 5:
            nummis = 5-len(rw)
            a_growths[rwi] = rw + rw[-nummis:]
    return a_growths


abkey = kylipair[np.where([spa in ky and spb in ky for ky in kylipair])][0]
ackey = kylipair[np.where([spa in ky and spc in ky for ky in kylipair])][0]
dfab = pair_data[abkey]
dfac = pair_data[ackey]



ab_grth = np.array(avg_growth_pair_data(dfab))
ac_grth = np.array(avg_growth_pair_data(dfac))



abmean = np.mean(ab_grth,axis = 0) - monomean
abstd = np.std(ab_grth,axis =0) - monostd

acmean = np.mean(ac_grth,axis = 0) - monomean
acstd = np.std(ac_grth,axis =0) - monostd


y3 = interp1d(x,abmean, kind = 'quadratic')
y4 = interp1d(x,abstd, kind = 'quadratic')

y5 = interp1d(x,acmean, kind = 'quadratic')
y6 = interp1d(x,acstd, kind = 'quadratic')



axs.plot(pltrng,y3(pltrng), color = '#E955E7', label = 'With '+spb, lw = 3)
axs.fill_between(pltrng,y3(pltrng)+abs(y4(pltrng)),y3(pltrng)-abs(y4(pltrng)),alpha = 0.3, color = '#E955E7', label = '_ab')

axs.plot(pltrng,y5(pltrng), color = '#CB9735', label = 'With ' + spc, lw = 3)
axs.fill_between(pltrng,y5(pltrng)+abs(y6(pltrng)),y5(pltrng)-abs(y6(pltrng)),alpha = 0.3, color = '#CB9735', label = '_ac')

axs.set_title('Deviation in per-organism growth rate of ' + spa)

axs.set_xlabel('Time', fontsize = 15)
axs.set_ylabel(r'$\Delta g_i^{\circ}$', fontsize = 15)

axs.set_ylim(-1,5)

axs.plot(pltrng,[0]*len(pltrng),':',color = 'k', lw = 3)

plt.legend()
plt.show()

fig.savefig('paperplots/'+spa)
