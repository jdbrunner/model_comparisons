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

from contextlib import redirect_stdout


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


mono_params_df = pd.read_csv('mono_parameters.csv',index_col  = 0)
pair_outs_fit = pd.read_csv('pair_outcomes.csv',index_col  = 0)
interact_params = pd.read_csv('lotka_volterra_fitted.csv',index_col  = 0)

air_outs_fit.index = [(iii.split("'")[1],iii.split("'")[3]) for iii in pair_outs_fit.index]


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






#test the parameters we have:
#From the paper:
correct_bool_gore, pair_summary_gore, trio_summary_gore, glv_details_gore = test_params_LV(interacts_gore, mono_params_gore['K'], mono_params_gore['r'], real_outs, pair_outs_gore)
#From the paper, with pairs corrected by observation
correct_bool_gore_pf, pair_summary_gore_pf, trio_summary_gore_pf, glv_details_gore_pf = test_params_LV(interacts_gore_pf, mono_params_gore['K'], mono_params_gore['r'], real_outs, pair_outs_gore)
#From my own fitting
correct_bool, pair_summary, trio_summary,glv_details = test_params_LV(interact_params, mono_params_df['K'], mono_params_df['r'],real_outs,pair_outs_fit)

trio_summary.loc[[('Pv','Pf','Pci')]]

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




full_attempt = find_full_sol(interact_params,mono_params_df['K'],mono_params_df['r'],trio_summary,pair_summary, fix_them,max_mints = 360)

correct_bool_searched, pair_summary_searched, trio_summary_searched,glv_details_searched = test_params_LV(full_attempt[1], mono_params_df['K'], mono_params_df['r'],real_outs,pair_outs_fit)


s4 = len(trio_summary_searched[trio_summary_searched.gLVRight == True])/len(trio_summary_searched)


print('Proportion of Correct Trios:')
print('From Friedman et al:', s1)
print('From Friedman et al, with pairs corrected:', s2)
print('From new parameter fitting:', s3)
print('From computational search:',s4)


pair_exp = False
if pair_exp:
    can_fix = {}
    for ky in fix_them[0].keys():
        can_fix[ky] = sum(list(it.chain(fix_them[0][ky].values())))/len(fix_them[0][ky].values())

    can_fix_df = pd.DataFrame.from_dict(can_fix, columns = ['Propotion Fixable'], orient = 'index')
    pair_labels = [sp[0]+', '+sp[1] for sp in can_fix_df.index]
    can_fix_df.index = pair_labels

    can_fix_df.to_pickle('pair_experiment')

    # can_fix_df = pd.read_pickle('pair_experiment')

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
