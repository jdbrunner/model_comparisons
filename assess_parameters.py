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
#I need to correct for 0s in my pair timecourse data.



# print('asses_parameters.py: estimating parameters')
# from estimate_parameters import *
#Creates a DataFrame with estimated single species parameters (mono_params_df)
#Creates a dictionary with pair data scaled by carrying capacity (pair_data_scld)
#Creates a dataframe with pair outcomes according to model fitting (pair_outs_fit)
#Creates a dataframe of fit pairwise interaction parameters (interact_params)

mono_params_df = pd.read_csv('mono_parameters.csv',index_col  = 0)
pair_outs_fit = pd.read_csv('pair_outcomes.csv',index_col  = 0)
interact_params = pd.read_csv('lotka_volterra_fitted.csv',index_col  = 0)
interact_params_search = pd.read_csv('lotka_volterra_search.csv',index_col  = 0)

pair_outs_fit.index = [(iii.split("'")[1],iii.split("'")[3]) for iii in pair_outs_fit.index]


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

print('asses_parameters.py: assessing parameter sets')
#test the parameters we have:
#From the paper:
correct_bool_gore, pair_summary_gore, trio_summary_gore, glv_details_gore = test_params_LV(interacts_gore, mono_params_gore['K'], mono_params_gore['r'], real_outs, pair_outs_gore)
#From the paper, with pairs corrected by observation
correct_bool_gore_pf, pair_summary_gore_pf, trio_summary_gore_pf, glv_details_gore_pf = test_params_LV(interacts_gore_pf, mono_params_gore['K'], mono_params_gore['r'], real_outs, pair_outs_gore)
#From my own fitting
correct_bool, pair_summary, trio_summary,glv_details = test_params_LV(interact_params, mono_params_df['K'], mono_params_df['r'],real_outs,pair_outs_fit)
#From my search
correct_bool_search, pair_summary_search, trio_summary_search,glv_details_search = test_params_LV(interact_params, mono_params_df['K'], mono_params_df['r'],real_outs,pair_outs_fit)

#Parameters found with computational search.
# found_params = pd.read_pickle()


###To see how well each of the above did:
#From the paper:
s1 = len(trio_summary_gore[trio_summary_gore.gLVRight == True])/len(trio_summary_gore)
#From the paper, with the pairs corrected:
s2 = len(trio_summary_gore_pf[trio_summary_gore_pf.gLVRight == True])/len(trio_summary_gore_pf)
#From my own fitting:
s3 = len(trio_summary[trio_summary.gLVRight == True])/len(trio_summary)
#From my search:
s4 = len(trio_summary[trio_summary_search.gLVRight == True])/len(trio_summary_search)

print('Proportion of Correct Trios:')
print('From Friedman et al:', s1)
print('From Friedman et al, with pairs corrected:', s2)
print('From new parameter fitting:', s3)
print('From computational search:',s4)



alpha_paramsets = {'FromFriedman':interacts_gore,'FriedmanFixed':interacts_gore_pf, 'Refitted':interact_params, 'Searched':interact_params_search}
mono_paramsets = {'FromFriedman':mono_params_gore,'FriedmanFixed':mono_params_gore, 'Refitted':mono_params_df, 'Searched':mono_params_df}


print('asses_parameters.py: generating plots')
for ky in alpha_paramsets.keys():
    pair_data_scld2 = {}

    for ii in pair_data.keys():
        pair_data_scld2[ii] = pair_data[ii].copy()
        spes = ii.split('_')
        pair_data_scld2[ii].loc[:,spes[0]] = pair_data_scld2[ii].loc[:,spes[0]]/mono_paramsets[ky].loc[spes[0],'K']
        pair_data_scld2[ii].loc[:,spes[1]] = pair_data_scld2[ii].loc[:,spes[1]]/mono_paramsets[ky].loc[spes[1],'K']


    for i in pair_data.keys():
        print(i)
        fig, axs = plt.subplots(1,4, figsize = (45,15),tight_layout = True)
        fig.suptitle(i,fontsize=30)
        spes = i.split('_')
        inits2 = pair_data_scld2[i].loc[1].loc[0,:].values
        axs[0].set_title('State Space')
        axs[2].set_title('State Space')
        axs[1].set_title('Time Course')
        axs[3].set_title('Time Course')
        print('rs:', [mono_paramsets[ky].loc[spes[0],'r'], mono_paramsets[ky].loc[spes[1],'r']])
        print('alphas:',[alpha_paramsets[ky].loc[spes[0],spes[1]],alpha_paramsets[ky].loc[spes[1],spes[0]]])
        for j in pair_data_scld2[i].index.levels[0]:
            if all(pair_data_scld2[i].loc[j].loc[0,:].values == inits2):
                ax3 = axs[0]
                ax4 = axs[1]
            else:
                ax3 = axs[2]
                ax4 = axs[3]
            # pair_data_scld2[i].loc[j].loc[:,spes[0]].plot(style = 'rx', ax = ax4,markersize=12)
            # pair_data_scld2[i].loc[j].loc[:,spes[1]].plot(style = 'bx', ax = ax4,markersize=12)
            # ax3.plot(pair_data_scld2[i].loc[j].loc[:,spes[0]],pair_data_scld2[i].loc[j].loc[:,spes[1]],'bx',linewidth=2, markersize=12)
            sols2,times2  = solve_vode(twospec_num,[[mono_paramsets[ky].loc[spes[0],'r'], mono_paramsets[ky].loc[spes[1],'r']],[alpha_paramsets[ky].loc[spes[0],spes[1]],alpha_paramsets[ky].loc[spes[1],spes[0]]]],pair_data_scld2[i].loc[j].loc[1,:].values,1000,t0 = 1)
            ax4.plot(times2,sols2[:,0],color = 'r',linewidth=4, markersize=12)
            ax4.plot(times2,sols2[:,1], color = 'b',linewidth=4, markersize=12)
            ax3.plot(sols2[:,0],sols2[:,1],color = 'b',linewidth=4, markersize=12)
            ax3.set_xlabel('')
            ax4.set_xlabel('')
        fig.savefig('parameter_fit_plots/PAIR/'+ky+'_'+i)
        plt.close()



    for spes in real_outs.index:
        rxnrates = make_k_trio(alpha_paramsets[ky],mono_paramsets[ky]['r'],spes)
        x0 = rand(3)
        endt = 1000
        sols,times  = solve_vode(tthreespec_num,[[mono_paramsets[ky].loc[spes[0],'r'],mono_paramsets[ky].loc[spes[1],'r'],mono_paramsets[ky].loc[spes[2],'r']],[alpha_paramsets[ky].loc[spes[0],spes[1]],alpha_paramsets[ky].loc[spes[0],spes[2]],alpha_paramsets[ky].loc[spes[1],spes[0]],alpha_paramsets[ky].loc[spes[1],spes[2]],alpha_paramsets[ky].loc[spes[2],spes[0]],alpha_paramsets[ky].loc[spes[2],spes[1]]]],x0,endt)
        realiz_fig,realiz_ax = plt.subplots(1,1,figsize = (15,15))
        realiz_fig.suptitle(spes, fontsize = 30)
        realiz_ax.set_title('Parameters Fitted to Pairs')
        realiz_ax.plot(times,sols[:,0],'r',times,sols[:,1], 'b',times,sols[:,2],'g',linewidth=4,)
        realiz_ax.legend(spes, fontsize = 40)
        realiz_fig.savefig('parameter_fit_plots/TRIO/'+ky+'_'+'_'.join(spes))
        plt.close()
