from os.path import expanduser, join
from os import makedirs
import pkg_resources

import pandas as pd
import numpy as np
import scipy as sp
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
#Creates a DataFrame of the observed outcomes of the trio experiments (real_outs)
#Creates a DataFrame of the observed outcomes of the pair experiment "by the eye test" (pair_outs_gore)
#Creates a DataFrame of the reported k&r values from the Gore paper (mono_params_gore)
#Creates a DataFrame of the reported parameters from the Gore paper (interacts_gore_unscld)
#Creates a DataFrame of the reported parameters after model rescaling (interacts_gore)
#Creates a DataFrame of the pair model analysis (pair_out_gore)
#Creates a DataFrame of the reported and rescaled parameters adjusted to match pair observations (interacts_gore_pf)
#Creates a DataFrame of the single species timecourse experiments (mono_data)
#Creates a DataFrame of the pair experiment timecourse data (pair_data)
#Creates a DataFrame of the trio experiment (only start/end) (trio_data)
#Creates a DataFrame of the group experiment data (only start/end) (groups_data)


exp_index = list(real_outs.index)

pair_data



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

def avg_xd(experim,tstp = 1):
    '''experim is an array of values representing a time course of biomass. Calculates the average growth rate'''
    #first filter the NaN
    experim = experim[np.invert(np.isnan(experim))]
    grates = np.array([experim[i+1] - experim[i] for i in range(len(experim)-1)])/tstp
    #avoid dividing by 0
    grates_per_org = np.array(grates)
    return np.mean(grates_per_org[grates_per_org != 0])

def avg_growth_mono_data(m_data):
    a_growths = np.array([avg_xdoverx(m_data[Col].values) for Col in m_data.columns])
    return np.mean(a_growths)

def avg_growth_mono_data2(m_data):
    a_growths = np.array([avg_xd(m_data[Col].values) for Col in m_data.columns])
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


# for ind in avg_growth_trio.index:
#     spa,spb,spc = ind
#     kyhere = kyli[np.where([spa in ky and spb in ky and spc in ky for ky in kyli])][0]
#     exp_df = trio_data[kyhere]
#     avg_growth = (np.mean((exp_df.xs(5.0,level = 1) - exp_df.xs(0.0,level = 1)))) [[spa,spb,spc]]
#     avg_growth_trio.loc[ind] = avg_growth.values



kylipair = np.array(list(pair_data.keys()))



def avg_growth_pair_data(p_data):
    exp_nums = p_data.index.levels[0]
    a_growths = np.array([[avg_xdoverx(p_data.loc[l][Col].values) for Col in p_data.columns] for l in exp_nums])
    return pd.Series(np.mean(a_growths, axis = 0),index = p_data.columns)

def avg_growth_pair_data2(p_data):
    exp_nums = p_data.index.levels[0]
    a_growths = np.array([[avg_xd(p_data.loc[l][Col].values) for Col in p_data.columns] for l in exp_nums])
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



species_nm_df = pd.DataFrame(index = exp_index, columns = ['Species_A','Species_B','Species_C'])
for ind in species_nm_df.index:
    species_nm_df.loc[ind] = list(ind)


avg_experimental_growth = pd.concat([species_nm_df,avg_growth_trio,avg_growth_pair,avg_growth_mono],axis = 1)

avg_experimental_growth.index = range(len(avg_experimental_growth))
did_add_yn_exp = pd.DataFrame(columns = ['Species 1','Species 2','Species 3','Change species 1','Change species 2','Change species 3'])
did_add_yn_exp_pic = pd.DataFrame(columns = [r'\Huge $e_{1|23}$',r'\Huge $e_{2|13}$',r'\Huge $e_{3|12}$'])


def delg(trio_rw):
    comps = np.array([[trio_rw.A_Together-trio_rw.A_Alone,0,trio_rw.A_with_B-trio_rw.A_Alone,trio_rw.A_with_C- trio_rw.A_Alone],[trio_rw.B_Together-trio_rw.B_Alone,trio_rw.B_with_A-trio_rw.B_Alone,0,trio_rw.B_with_C- trio_rw.B_Alone],[trio_rw.C_Together-trio_rw.C_Alone,trio_rw.C_with_A-trio_rw.C_Alone,trio_rw.C_with_B- trio_rw.C_Alone,0]])
    all_3 = trio_rw.Species_A + 'x' + trio_rw.Species_B + 'x' + trio_rw.Species_C
    labs_tub = [(all_3, trio_rw.Species_A), (all_3,trio_rw.Species_B),(all_3, trio_rw.Species_C)]
    index = pd.MultiIndex.from_tuples(labs_tub, names=['Trio', 'Microbe'])
    return pd.DataFrame(comps,index = index, columns = ['w/Both', 'w/' + trio_rw.Species_A, 'w/' + trio_rw.Species_B,'w/' + trio_rw.Species_C])


def delg2(trio_rw):
    comps = np.array([[trio_rw.A_Together-trio_rw.A_Alone,trio_rw.A_with_B-trio_rw.A_Alone,trio_rw.A_with_C- trio_rw.A_Alone],[trio_rw.B_Together-trio_rw.B_Alone,trio_rw.B_with_A-trio_rw.B_Alone,trio_rw.B_with_C- trio_rw.B_Alone],[trio_rw.C_Together-trio_rw.C_Alone,trio_rw.C_with_A-trio_rw.C_Alone,trio_rw.C_with_B- trio_rw.C_Alone]])
    all_3 = trio_rw.Species_A + 'x' + trio_rw.Species_B + 'x' + trio_rw.Species_C
    return pd.DataFrame(comps,index = [trio_rw.Species_A,trio_rw.Species_B,trio_rw.Species_C], columns = ['w/Both', 'w/1' , 'w/2' ])

deltags = delg(avg_experimental_growth.loc[0])
for grw in avg_experimental_growth.index[1:]:
    deltags = deltags.append(delg(avg_experimental_growth.loc[grw]))



prob_list = {}
for grw in avg_experimental_growth.index:
    df = delg2(avg_experimental_growth.loc[grw])
    for sp in df.index:
        pm = (df.loc[sp,'w/Both']>0)
        if pm:
            yn = (df.loc[sp,'w/1'] < 0 and df.loc[sp,'w/2'] <0)
        else:
            yn = (df.loc[sp,'w/1'] > 0 and df.loc[sp,'w/2'] >0)
        if yn:
            prob_list[grw] = df
            break


len(prob_list)


prob_list.keys()

delg(avg_experimental_growth.loc[26])
delg(avg_experimental_growth.loc[0])

len(avg_experimental_growth)
avg_experimental_growth


len(kylipair)


avg_growth_pair_data(pair_data[kylipair[0]])['Ea'] -avg_growth_mono_data(mono_data['Ea'])


estimated_interactions = pd.DataFrame(columns = ['Source','Target','Wght','Qual','AbsWeight'])
for pr in kylipair:
    prl = pr.split('_')
    pair_grth = avg_growth_pair_data(pair_data[pr])
    sp1g = avg_growth_mono_data(mono_data[prl[0]])
    sp2g = avg_growth_mono_data(mono_data[prl[1]])
    estimated_interactions = estimated_interactions.append(pd.DataFrame([[prl[0],prl[1],pair_grth[prl[0]]-sp1g,np.sign(pair_grth[prl[0]]-sp1g),abs(pair_grth[prl[0]]-sp1g)],[prl[1],prl[0],pair_grth[prl[1]]-sp2g,np.sign(pair_grth[prl[1]]-sp2g),abs(pair_grth[prl[1]]-sp2g)]], columns = estimated_interactions.columns))


estimated_interactions.to_csv('estimated.csv')
