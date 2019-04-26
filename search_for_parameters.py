#Script to search for parameters that match the correct outcomes.

from lv_pair_trio_functs import *
import numpy as np
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.integrate import ode
from numpy.random import rand

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

from estimate_parameters import *
#CL argument is whether or not to run in ``testing mode"
#Creates a DataFrame with estimated single species parameters (mono_params_df)
#Creates a dictionary with pair data scaled by carrying capacity (pair_data_scld)
#Creates a dataframe with pair outcomes according to model fitting (pair_outs_fit)
#Creates a dataframe of fit pairwise interaction parameters (interact_params)



#test the parameters we have:
#From the paper:
correct_bool_gore, pair_summary_gore, trio_summary_gore, glv_details_gore = test_params_LV(interacts_gore, mono_params_gore['K'], mono_params_gore['r'], real_outs, pair_outs_gore)
#From the paper, with pairs corrected by observation
correct_bool_gore_pf, pair_summary_gore_pf, trio_summary_gore_pf, glv_details_gore_pf = test_params_LV(interacts_gore_pf, mono_params_gore['K'], mono_params_gore['r'], real_outs, pair_outs_gore)
#From my own fitting
correct_bool, pair_summary, trio_summary,glv_details = test_params_LV(interact_params, mono_params_df['K'], mono_params_df['r'],real_outs,pair_outs_fit)







###To see how well each of the above did:
#From the paper:
s1 = len(trio_summary_gore[trio_summary_gore.gLVRight == True])/len(trio_summary_gore)
#From the paper, with the pairs corrected:
s2 = len(trio_summary_gore_pf[trio_summary_gore_pf.gLVRight == True])/len(trio_summary_gore_pf)
#From my own fitting:
s3 = len(trio_summary[trio_summary.gLVRight == True])/len(trio_summary)

#rank the pairs by how far their behavior in trios is from what gLV predicts
pair_problems = pd.Series(index = pair_summary.index)
for pr in pair_problems.index:
    pair_problems.loc[[pr]] = sum(trio_summary.loc[[set(pr).issubset(ti) for ti in trio_summary.index],'gLVRight'])
pair_problems.sort_values(inplace = True, ascending = False)

#iterating through all pairs, try to fix the pair without changing the pair's internal parameters.
fix_them = fix_all_pairs_gen(pair_problems,interact_params,mono_params_df['K'],mono_params_df['r'],pair_summary,trio_summary, how_long_to_wait = 360)
can_fix = {}
for ky in fix_them[0].keys():
    can_fix[ky] = sum(list(it.chain(fix_them[0][ky].values())))/len(fix_them[0][ky].values())




full_attempt = find_full_sol(interact_params,mono_params_df['K'],mono_params_df['r'],trio_summary,pair_summary, fix_them,max_mints = 360)

correct_bool_searched, pair_summary_searched, trio_summary_searched,glv_details_searched = test_params_LV(full_attempt[1], mono_params_df['K'], mono_params_df['r'],real_outs,pair_outs_fit)


s4 = len(trio_summary_searched[trio_summary_searched.gLVRight == True])/len(trio_summary_searched)



can_fix_df = pd.DataFrame.from_dict(can_fix, columns = ['Propotion Fixable'], orient = 'index')
pair_labels = [sp[0]+', '+sp[1] for sp in can_fix_df.index]
can_fix_df.index = pair_labels

can_fix_df.to_pickle('pair_experiment')


import seaborn as sn

sn.set(font_scale=2)
fix_fig1,fix_ax1 = plt.subplots(1,1,figsize = (20,2))
fix_ax1.tick_params(labelsize = 13)
sn.heatmap(can_fix_df.iloc[:14].T, annot = True, ax = fix_ax1, cbar = False, linewidths = 0.1, linecolor = 'white', yticklabels = False, cmap = 'YlGnBu', vmin = 0, vmax = 1)
fix_fig1.tight_layout()
fix_fig1.savefig('../search_result1')
fix_fig2,fix_ax2 = plt.subplots(1,1,figsize = (20,2))
fix_ax2.tick_params(labelsize = 13)
sn.heatmap(can_fix_df.iloc[14:].T, annot = True, ax = fix_ax2, cbar = False, linewidths = 0.1, linecolor = 'white', yticklabels = False, cmap = 'YlGnBu', vmin = 0, vmax = 1)
fix_fig2.tight_layout()
fix_fig2.savefig('../search_result2')
###################

pair_data_scld2 = {}
for ii in pair_data.keys():
    pair_data_scld2[ii] = pair_data[ii].copy()
    spes = ii.split('_')
    pair_data_scld2[ii].loc[:,spes[0]] = pair_data_scld2[ii].loc[:,spes[0]]/mono_params_gore.loc[spes[0],'K']
    pair_data_scld2[ii].loc[:,spes[1]] = pair_data_scld2[ii].loc[:,spes[1]]/mono_params_gore.loc[spes[1],'K']



for i in pair_data.keys():
    fig, axs = plt.subplots(1,4, figsize = (45,15),tight_layout = True)
    fig.suptitle(i,fontsize=30)
    spes = i.split('_')
    inits1 = pair_data_scld2[i].loc[1].loc[0,:].values
    inits2 = pair_data_scld[i].loc[1].loc[0,:].values
    # axs[0,0].set_title('Parameters from Friedman, Higgens, & Gore (2017)')
    # axs[0,1].set_title('Parameters from Friedman, Higgens, & Gore (2017)')
    # axs[1,0].set_title('Parameters from Friedman, Higgens, & Gore (2017)')
    # axs[1,1].set_title('Parameters from Friedman, Higgens, & Gore (2017)')
    axs[0].set_title('State Space')
    axs[2].set_title('State Space')
    axs[1].set_title('Time Course')
    axs[3].set_title('Time Course')
    for j in pair_data_scld2[i].index.levels[0]:
        # if all(pair_data_scld2[i].loc[j].loc[0,:].values == inits1):
        #     ax1 = axs[0,0]
        #     ax2 = axs[1,0]
        # else:
        #     ax1 = axs[0,1]
        #     ax2 = axs[1,1]
        # ax1.plot(pair_data_scld2[i].loc[j].loc[:,spes[0]],pair_data_scld2[i].loc[j].loc[:,spes[1]], 'bx')
        # pair_data_scld2[i].loc[j].loc[:,spes[0]].plot(style = 'rx', ax = ax2)
        # pair_data_scld2[i].loc[j].loc[:,spes[1]].plot(style = 'gx', ax = ax2)
        # sols,times  = solve_vode(twospec_num,[[mono_params_gore.loc[spes[0],'r'],mono_params_gore.loc[spes[1],'r']],[interacts_gore.loc[spes[0],spes[1]],interacts_gore.loc[spes[1],spes[0]]]],pair_data_scld2[i].loc[j].loc[0,:].values,5)
        # ax2.plot(times,sols[:,0],color = 'r')
        # ax2.plot(times,sols[:,1], color = 'g')
        # ax1.plot(sols[:,0],sols[:,1],color = 'b')
        # ax1.set_xlabel('')
        # ax2.set_xlabel('')
        if all(pair_data_scld[i].loc[j].loc[0,:].values == inits2):
            ax3 = axs[0]
            ax4 = axs[1]
        else:
            ax3 = axs[2]
            ax4 = axs[3]
        pair_data_scld[i].loc[j].loc[:,spes[0]].plot(style = 'rx', ax = ax4,markersize=12)
        pair_data_scld[i].loc[j].loc[:,spes[1]].plot(style = 'bx', ax = ax4,markersize=12)
        ax3.plot(pair_data_scld[i].loc[j].loc[:,spes[0]],pair_data_scld[i].loc[j].loc[:,spes[1]],'bx',linewidth=2, markersize=12)
        sols2,times2  = solve_vode(twospec_num,[[mono_params_df.loc[spes[0],'r'], mono_params_df.loc[spes[1],'r']],inter_params_dict[tuple(spes)]],pair_data_scld[i].loc[j].loc[1,:].values,5,t0 = 1)
        ax4.plot(times2,sols2[:,0],color = 'r',linewidth=4, markersize=12)
        ax4.plot(times2,sols2[:,1], color = 'b',linewidth=4, markersize=12)
        ax3.plot(sols2[:,0],sols2[:,1],color = 'b',linewidth=4, markersize=12)
        ax3.set_xlabel('')
        ax4.set_xlabel('')
    fig.savefig('../parameter_fit_plots/'+i+'row')



for spes in real_outs.index:
    rxnrates = make_k_trio(interact_params,mono_params_df['r'],spes)
    x0 = rand(3)
    endt = 100
    sols,times  = solve_vode(tthreespec_num,[[mono_params_df.loc[spes[0],'r'],mono_params_df.loc[spes[1],'r'],mono_params_df.loc[spes[2],'r']],[interact_params.loc[spes[0],spes[1]],interact_params.loc[spes[0],spes[2]],interact_params.loc[spes[1],spes[0]],interact_params.loc[spes[1],spes[2]],interact_params.loc[spes[2],spes[0]],interact_params.loc[spes[2],spes[1]]]],x0,endt)
    realiz_fig,realiz_ax = plt.subplots(1,1,figsize = (15,15))
    realiz_fig.suptitle(spes, fontsize = 30)
    realiz_ax.set_title('Parameters Fitted to Pairs')
    realiz_ax.plot(times,sols[:,0],'r',times,sols[:,1], 'b',times,sols[:,2],'g',linewidth=4,)
    realiz_ax.legend(spes, fontsize = 40)
    realiz_fig.savefig('../parameter_fit_plots/'+'_'.join(spes))
