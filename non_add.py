import mminte
from os.path import expanduser, join
from os import makedirs
import pkg_resources

import pandas as pd
import numpy as np
import scipy as sp
import cobra as cb
import itertools

import matplotlib.pyplot as plt
import seaborn as sb


import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.integrate import ode
from numpy.random import rand
from lv_pair_trio_functs import *



def make_hmp(trio_rw):
    df = pd.DataFrame(-np.array([[trio_rw.A_Together-trio_rw.A_Alone,trio_rw.A_Together-trio_rw.A_Alone,trio_rw.B_Together-trio_rw.B_Alone,trio_rw.B_Together-trio_rw.B_Alone,trio_rw.C_Together-trio_rw.C_Alone,trio_rw.C_Together-trio_rw.C_Alone],[trio_rw.A_with_B-trio_rw.A_Alone, trio_rw.A_with_C- trio_rw.A_Alone,trio_rw.B_with_A-trio_rw.B_Alone, trio_rw.B_with_C- trio_rw.B_Alone,trio_rw.C_with_A-trio_rw.C_Alone, trio_rw.C_with_B- trio_rw.C_Alone]]))
    Xlabels = [trio_rw.Species_A + '('+trio_rw.Species_B + ')',trio_rw.Species_A + '('+trio_rw.Species_C + ')',trio_rw.Species_B + '('+trio_rw.Species_A + ')',trio_rw.Species_B + '('+trio_rw.Species_C + ')',trio_rw.Species_C + '('+trio_rw.Species_A + ')',trio_rw.Species_C + '('+trio_rw.Species_B + ')']
    return df,Xlabels

def did_add(trio_rw):
    pair_tri = np.array([[trio_rw.A_Together-trio_rw.A_Alone,trio_rw.B_Together-trio_rw.B_Alone,trio_rw.C_Together-trio_rw.C_Alone],[(trio_rw.A_with_B-trio_rw.A_Alone) + (trio_rw.A_with_C- trio_rw.A_Alone),(trio_rw.B_with_A-trio_rw.B_Alone)+ (trio_rw.B_with_C- trio_rw.B_Alone),(trio_rw.C_with_A-trio_rw.C_Alone) + (trio_rw.C_with_B- trio_rw.C_Alone)]])
    comp = ((pair_tri[1]-pair_tri[0]))/abs(pair_tri[0])
    Xlabels = [trio_rw.Species_A ,trio_rw.Species_B ,trio_rw.Species_C ]
    df = pd.DataFrame([Xlabels + list(comp)],columns = ['Species 1','Species 2','Species 3','Change species 1','Change species 2','Change species 3'])
    df_pic = pd.DataFrame([comp],columns = [r'\Huge $e_{1|23}$',r'\Huge $e_{2|13}$',r'\Huge $e_{3|12}$'],index = [trio_rw.Species_A  + ', '+ trio_rw.Species_B + ', '+ trio_rw.Species_C])
    return df,df_pic



from load_gore_results import *



analysis_folder = join(expanduser('~'), 'Documents/community_vs_lv/metabolic_networks')
exp_index = list(real_outs.index)



ser = mono_data['Pch']['A'].values
ser[np.invert(np.isnan(ser))]



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
    return np.mean(grates_per_org[grates_per_org != 0])

def avg_growth_mono_data(m_data):
    a_growths = np.array([avg_xdoverx(m_data[Col].values) for Col in m_data.columns])
    return np.mean(a_growths)



avg_growth_mono = pd.DataFrame(index = exp_index, columns = ['A_Alone','B_Alone','C_Alone'])
for ind in avg_growth_mono.index:
    spa,spb,spc = ind
    avg_growth_mono.loc[ind] = [avg_growth_mono_data(mono_data[spa]),avg_growth_mono_data(mono_data[spb]),avg_growth_mono_data(mono_data[spc])]


kyli = np.array(list(trio_data.keys()))
avg_growth_trio = pd.DataFrame(index = exp_index, columns = ['A_Together','B_Together','C_Together'])
for ind in avg_growth_trio.index:
    spa,spb,spc = ind
    kyhere = kyli[np.where([spa in ky and spb in ky and spc in ky for ky in kyli])][0]
    exp_df = trio_data[kyhere]
    avg_growth = (np.mean((exp_df.xs(5.0,level = 1) - exp_df.xs(0.0,level = 1))/(5*exp_df.xs(0.0,level = 1))))[[spa,spb,spc]]
    avg_growth_trio.loc[ind] = avg_growth.values





kylipair = np.array(list(pair_data.keys()))



def avg_growth_pair_data(p_data):
    exp_nums = p_data.index.levels[0]
    a_growths = np.array([[avg_xdoverx(p_data.loc[l][Col].values) for Col in p_data.columns] for l in exp_nums])
    return pd.Series(np.mean(a_growths, axis = 0),index = p_data.columns)




avg_growth_pair = pd.DataFrame(index = exp_index, columns = ['A_with_B','A_with_C','B_with_A','B_with_C','C_with_A','C_with_B'])
for ind in avg_growth_pair.index:
    spa,spb,spc = ind
    abkey = kylipair[np.where([spa in ky and spb in ky for ky in kylipair])][0]
    ackey = kylipair[np.where([spa in ky and spc in ky for ky in kylipair])][0]
    bckey = kylipair[np.where([spb in ky and spc in ky for ky in kylipair])][0]
    dfab = pair_data[abkey]
    dfac = pair_data[ackey]
    dfbc = pair_data[bckey]
    ab_grth = avg_growth_pair_data(dfab)
    ac_grth = avg_growth_pair_data(dfac)
    bc_grth = avg_growth_pair_data(dfbc)
    rw = [ab_grth.loc[spa],ac_grth.loc[spa],ab_grth.loc[spb],bc_grth.loc[spb],ac_grth.loc[spc],bc_grth.loc[spc]]
    avg_growth_pair.loc[ind] = rw

avg_growth_pair
species_nm_df = pd.DataFrame(index = exp_index, columns = ['Species_A','Species_B','Species_C'])
for ind in species_nm_df.index:
    species_nm_df.loc[ind] = list(ind)


avg_experimental_growth = pd.concat([species_nm_df,avg_growth_trio,avg_growth_pair,avg_growth_mono],axis = 1)

avg_experimental_growth.index = range(len(avg_experimental_growth))
did_add_yn_exp = pd.DataFrame(columns = ['Species 1','Species 2','Species 3','Change species 1','Change species 2','Change species 3'])
did_add_yn_exp_pic = pd.DataFrame(columns = [r'\Huge $e_{1|23}$',r'\Huge $e_{2|13}$',r'\Huge $e_{3|12}$'])


for grw in avg_experimental_growth.index:
    v1,v2 = did_add(avg_experimental_growth.loc[grw])
    did_add_yn_exp = did_add_yn_exp.append(v1)
    did_add_yn_exp_pic = did_add_yn_exp_pic.append(v2)

did_add_yn_exp_pic

fig,ax1 = plt.subplots(1,figsize=(25,5),tight_layout = True)
sb.set(font_scale=2)
plt.rc('text', usetex=True)
sb.heatmap(did_add_yn_exp_pic.T,cmap = 'YlGnBu',center = 0, cbar = True, ax = ax1,annot= False, robust = True,linewidths = 0.1, linecolor = 'white',)# vmin = -10,vmax = 10)
#fig.savefig(join(analysis_folder, 'mminte_hmps/add_exp_fldchng'))


sum(did_add_yn_exp_pic.T.describe().loc['min'] < -1)
sum(did_add_yn_exp_pic.T.describe().loc['max'] > 1)

sum(np.logical_or((did_add_yn_exp_pic.T.describe().loc['min'] < -5).values,(did_add_yn_exp_pic.T.describe().loc['max'] > 5).values))


did_add_yn_exp.index = range(len(did_add_yn_exp))
did_add_yn_exp_pic.T.describe().loc['mean']
did_add_yn_exp_pic.values.flatten()
did_add_series = pd.Series(did_add_yn_exp_pic.values.flatten())
did_add_series.describe()

from scipy import stats
did_add_yn_exp_pic[(np.abs(stats.zscore(did_add_yn_exp_pic)) < 3).all(axis=1)]




fig2,ax12 = plt.subplots(1,figsize=(20,5),tight_layout = True)
sb.set(font_scale=2)
sb.distplot(did_add_series)
#fig2.savefig(join(analysis_folder, 'mminte_hmps/add_exp_fldchng_dist'))


def make_add_comp(trio_rw):
    df = pd.DataFrame(np.array([[trio_rw.A_Together-trio_rw.A_Alone,trio_rw.B_Together-trio_rw.B_Alone,trio_rw.C_Together-trio_rw.C_Alone],[trio_rw.A_with_B-trio_rw.A_Alone + trio_rw.A_with_C- trio_rw.A_Alone,trio_rw.B_with_A-trio_rw.B_Alone+ trio_rw.B_with_C- trio_rw.B_Alone,trio_rw.C_with_A-trio_rw.C_Alone+ trio_rw.C_with_B- trio_rw.C_Alone]])).T
    df.index = [trio_rw.Species_A+'x'+'('+ trio_rw.Species_B+'x'+trio_rw.Species_C+')',trio_rw.Species_B+'x'+'('+ trio_rw.Species_A+'x'+trio_rw.Species_C+')',trio_rw.Species_C+'x'+'('+ trio_rw.Species_A+'x'+trio_rw.Species_B+')' ]
    df.columns = ['Trio','Sum_of_Pairs']
    return df

comp_infl = pd.DataFrame(columns = ['Trio','Sum_of_Pairs'])
for ind in avg_experimental_growth.index:
    comp_infl = comp_infl.append(make_add_comp(avg_experimental_growth.loc[ind]))


comp_infl['Diff'] = comp_infl['Trio'] - comp_infl['Sum_of_Pairs']

print(sp.stats.wilcoxon(comp_infl['Diff']))
