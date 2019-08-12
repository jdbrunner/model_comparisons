#script to estimate the parameters for the gLV models from the gore data
#Creates a DataFrame with estimated single species parameters (mono_params_df)
#Creates a dictionary with pair data scaled by carrying capacity (pair_data_scld)
#Creates a dataframe with pair outcomes according to model fitting (pair_outs_fit)
#Creates a dataframe of fit pairwise interaction parameters (interact_params)

#If running directly, for testing purposes:
# %run lv_pair_trio_functs
# #Running the module also loads:
# # import numpy as np
# # import itertools as it
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from scipy.optimize import minimize
# # from scipy.optimize import curve_fit
# # from scipy.optimize import least_squares
# # from scipy.integrate import ode
# # from numpy.random import rand
#
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
from numpy.random import rand
from lv_pair_trio_functs import *
from load_gore_results import *
import sys


nt = 100

mono_params = {}
for i in mono_data.keys():
    mono_params[i] = logistic_fit(mono_data[i])

mono_params_df = pd.DataFrame.from_dict(mono_params, columns = ['r','K'], orient = 'index')
#mono_params_df
#How different are mine from the paper?
#(mono_params_gore - mono_params_df)/mono_params_df
#Very ^

#adjust for very small carrying capacity
for ki in mono_params_df.index:
    if mono_params_df.loc[ki,'K'] < 0.01:
        mono_params_df.loc[ki,'K'] = 0.01

#Plot of the fitted curves
show_me_2 = False
if show_me_2:
    for i in mono_data.keys():
        fig, ax = plt.subplots()
        ax.set_title(i)
        ax.plot(mono_data[i],'*')
        C = (mono_params_df.loc[i,'K'] - mono_data[i].mean(axis = 1)[1])/mono_data[i].mean(axis = 1)[1]
        ax.plot(np.linspace(1,5,100),logistic_fun((np.linspace(1,5,100),mono_data[i].mean(axis = 1)[1]),mono_params_df.loc[i,'r'],C))

#I need to correct for 0s in my pair timecourse data.
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

inter_params_dict = {}
pair_outcome_dict = {}
for k in pair_data_scld.keys():#our own parameter fitting
    spes = k.split('_')
    pars,ou = model_fit_wrapper(pair_data_scld[k], [mono_params_df.loc[spes[0],'r'], mono_params_df.loc[spes[1],'r']], numtris = nt)
    pair_outcome_dict[tuple(spes)] = '-'.join(np.array(spes)[~ou])
    inter_params_dict[tuple(spes)] = pars


#pair_outcome_dict
pair_outs_fit = pd.DataFrame.from_dict(pair_outcome_dict, orient = 'index')
pair_outs_fit = pair_outs_fit.rename({0:'Observed'}, axis = 1)
all_spec = mono_data.keys()
interact_params = pd.DataFrame(np.eye(len(all_spec)),index = all_spec,columns = all_spec)
for pr in inter_params_dict.keys():
    interact_params.loc[pr[0],pr[1]] = inter_params_dict[pr][0]
    interact_params.loc[pr[1],pr[0]] = inter_params_dict[pr][1]


#Creates a DataFrame with estimated single species parameters (mono_params_df)
#Creates a dictionary with pair data scaled by carrying capacity (pair_data_scld)
#Creates a dataframe with pair outcomes according to model fitting (pair_outs_fit)
#Creates a dataframe of fit pairwise interaction parameters (interact_params)
mono_params_df.to_csv('mono_parameters_n.csv')
pair_outs_fit.to_csv('pair_outcomes_n.csv')
interact_params.to_csv('lotka_volterra_fitted_n.csv')




show_me = False
if show_me:#plotting of system with fitted parameters

    rvals = pd.Series({'Ea':0.46, 'Pa':0.55,'Pch':0.18,'Pci':0.16,'Pf':0.25,'Pp':0.65,'Pv':0.57,'Sm':0.34})
    kvals = pd.Series({'Ea':0.13, 'Pa':0.07,'Pch':0.11,'Pci':0.01,'Pf':0.05,'Pp':0.14,'Pv':0.11,'Sm':0.15})



    indx = ['Ea','Pa','Pch','Pci','Pf','Pp','Pv','Sm']
    inter = np.array([[1,0.69,1.09,0.55,1.53,0.82,1.09,0.72],[-0.18,1,2.44,-2.58,1.13,0.43,0.01,0.21],[-0.11,-0.8,1,-15.75,0.29,-0.04,-0.05,-0.03],[-0.32,0,0.18,1,-3.39,0,0.05,-0.3],[-0.02,0.28,1.2,0.83,1,0.01,0.07,-0.1],[0.87,1.58,1.24,0.24,1,1,1.01,0.84],[0.83,0.28,0.47,0,-0.02,0.79,1,0.7],[0.96,1.23,1.42,1.21,1.31,0.91,0.98,1]])#Parameters reported in Friedman et al.


    interacts = pd.DataFrame(inter,index = indx,columns = indx)
    #interacts
    interacts2 = -interacts*np.outer(1/kvals,kvals)

    pair_data_scld2 = {}

    for ii in pair_data.keys():
        pair_data_scld2[ii] = pair_data[ii].copy()
        spes = ii.split('_')
        pair_data_scld2[ii].loc[:,spes[0]] = pair_data_scld2[ii].loc[:,spes[0]]/kvals[spes[0]]
        pair_data_scld2[ii].loc[:,spes[1]] = pair_data_scld2[ii].loc[:,spes[1]]/kvals[spes[1]]

    for i in pair_data.keys():
        fig, axs = plt.subplots(4,2, figsize = (15,15))
        fig.suptitle(i)
        spes = i.split('_')
        inits1 = pair_data_scld2[i].loc[1].loc[0,:].values
        inits2 = pair_data_scld[i].loc[1].loc[0,:].values
        axs[0,0].set_title('Gore')
        axs[0,1].set_title('Gore')
        axs[1,0].set_title('Gore')
        axs[1,1].set_title('Gore')
        axs[2,0].set_title('Brunner')
        axs[2,1].set_title('Brunner')
        axs[3,0].set_title('Brunner')
        axs[3,1].set_title('Brunner')
        for j in pair_data_scld2[i].index.levels[0]:
            if all(pair_data_scld2[i].loc[j].loc[0,:].values == inits1):
                ax1 = axs[0,0]
                ax2 = axs[1,0]
            else:
                ax1 = axs[0,1]
                ax2 = axs[1,1]
            ax1.plot(pair_data_scld2[i].loc[j].loc[:,spes[0]],pair_data_scld2[i].loc[j].loc[:,spes[1]], 'bx')
            pair_data_scld2[i].loc[j].loc[:,spes[0]].plot(style = 'rx', ax = ax2)
            pair_data_scld2[i].loc[j].loc[:,spes[1]].plot(style = 'gx', ax = ax2)
            sols,times  = solve_vode(twospec_num,[[rvals[spes[0]],rvals[spes[1]]],[interacts2.loc[spes[0],spes[1]],interacts2.loc[spes[1],spes[0]]]],pair_data_scld2[i].loc[j].loc[0,:].values,5)
            ax2.plot(times,sols[:,0],color = 'r')
            ax2.plot(times,sols[:,1], color = 'g')
            ax1.plot(sols[:,0],sols[:,1],color = 'b')
            ax1.set_xlabel('')
            ax2.set_xlabel('')
            if all(pair_data_scld[i].loc[j].loc[0,:].values == inits2):
                ax3 = axs[2,0]
                ax4 = axs[3,0]
            else:
                ax3 = axs[2,1]
                ax4 = axs[3,1]
            pair_data_scld[i].loc[j].loc[:,spes[0]].plot(style = 'rx', ax = ax4)
            pair_data_scld[i].loc[j].loc[:,spes[1]].plot(style = 'gx', ax = ax4)
            ax3.plot(pair_data_scld[i].loc[j].loc[:,spes[0]],pair_data_scld[i].loc[j].loc[:,spes[1]],'bx')
            sols2,times2  = solve_vode(twospec_num,[[mono_params_df.loc[spes[0],'r'], mono_params_df.loc[spes[1],'r']],inter_params_dict[tuple(spes)]],pair_data_scld[i].loc[j].loc[1,:].values,5,t0 = 1)
            ax4.plot(times2,sols2[:,0],color = 'r')
            ax4.plot(times2,sols2[:,1], color = 'g')
            ax3.plot(sols2[:,0],sols2[:,1],color = 'b')
            ax3.set_xlabel('')
            ax4.set_xlabel('')

    pair_summary
    for spes in real_outs.index:
        rxnrates = make_k_trio(interact_params,mono_params_df['r'],spes)
        x0 = rand(3)
        endt = 100
        sols,times  = solve_vode(tthreespec_num,[[mono_params_df.loc[spes[0],'r'],mono_params_df.loc[spes[1],'r'],mono_params_df.loc[spes[2],'r']],[interact_params.loc[spes[0],spes[1]],interact_params.loc[spes[0],spes[2]],interact_params.loc[spes[1],spes[0]],interact_params.loc[spes[1],spes[2]],interact_params.loc[spes[2],spes[0]],interact_params.loc[spes[2],spes[1]]]],x0,endt)
        realiz_fig,realiz_ax = plt.subplots(1,1,figsize = (15,15))
        realiz_fig.suptitle(spes)
        realiz_ax.set_title('Newly Fitted Parameters')
        realiz_ax.plot(times,sols[:,0],'r:',times,sols[:,1], 'b:',times,sols[:,2],':g')
