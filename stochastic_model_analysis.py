#Script to test the probabilities of different outcomes using the stochastic gLV model, tested with Monte Carlo simulation.

#The stochastic model is
#X_i(t) = X(0) + Y_{i1}(\int(r_i X_i)) - Y_{i2}(\int(r_iX_i^2)) + \sum_{j\neq i}Y_{i}^j(\int(r_i \beta_{ij}X_iX_j))
#where the Y_i are unit rate Poisson processes.

#If running directly:
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
from numpy.random import binomial
from numpy.random import poisson

from load_gore_results import*
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
#Creates dictionary of the names of the species.


from estimate_parameters import*
#CL argument is whether or not to run in "test mode".
#Creates a DataFrame with estimated single species parameters (mono_params_df)
#Creates a dictionary with pair data scaled by carrying capacity (pair_data_scld)
#Creates a dataframe with pair outcomes according to model fitting (pair_outs_fit)
#Creates a dataframe of fit pairwise interaction parameters (interact_params)


#As a preliminary bit of fun, we may simulate sample paths. For this, we need to define our ``source" and ``products", because Gillespie's algorithm wants to treat this network as a reaction network.
#2 Species Model
two_spec_sources = np.array([[1,0],[2,0],[0,1],[0,2],[1,1],[1,1],[1,1],[1,1]])
two_spec_prodcts = np.array([[2,0],[1,0],[0,2],[0,1],[2,1],[0,1],[1,2],[1,0]])
#Three Species Model
three_spec_sources = np.array([[1,0,0],[2,0,0],[0,1,0],[0,2,0],[0,0,1],[0,0,2],[1,1,0],[1,1,0],[1,0,1],[1,0,1],[1,1,0],[1,1,0],[0,1,1],[0,1,1],[1,0,1],[1,0,1],[0,1,1],[0,1,1]])
three_spec_prodcts = np.array([[2,0,0],[1,0,0],[0,2,0],[0,1,0],[0,0,2],[0,0,1],[2,1,0],[0,1,0],[2,0,1],[0,0,1],[1,2,0],[1,0,0],[0,2,1],[0,0,1],[1,0,2],[1,0,0],[0,1,2],[0,1,0]])


#now, we generate a sample path and the solution to the deterministic ODE for each pair, mainly to produce fun pictures.

#for the pair model
for spes in pair_outs_fit.index:
    rxnrates = make_k_pair(interact_params,mono_params_df['r'],spes)
    x0 = rand(2)#take a random initial condition.
    endt = 20
    stoch = gillep(two_spec_sources,two_spec_prodcts,rxnrates,x0,endt,N = 100)#N is the ``volume" paramter
    #also generate a solution to the deterministic ODE
    sols,times  = solve_vode(twospec_num,[[mono_params_df.loc[spes[0],'r'],mono_params_df.loc[spes[1],'r']],[interact_params.loc[spes[0],spes[1]],interact_params.loc[spes[1],spes[0]]]],x0,endt)
    #and plot.
    realiz_fig,realiz_ax = plt.subplots(1,1,figsize = (15,5))
    realiz_fig.suptitle(spes[0]+' & '+spes[1],fontsize = 20)
    realiz_ax.set_title('Pair Fitted Parameters')
    realiz_ax.set_xlabel('time')
    realiz_ax.set_ylabel('population size')
    realiz_ax.step(stoch[0],stoch[1][:,0],'r',label = spes[0])
    realiz_ax.step(stoch[0],stoch[1][:,1],'b',label = spes[1])
    realiz_ax.plot(times,sols[:,0],'r:',times,sols[:,1], 'b:')
    realiz_ax.legend(fontsize = 20)
    #also, we do the same for the Paper's reported parameters.
    # rxnrates2 = make_k_pair(interacts_gore,mono_params_gore['r'],spes)
    # stoch2 = gillep(two_spec_sources,two_spec_prodcts,rxnrates2,x0,endt,N = 100)
    # sols2,times2  = solve_vode(twospec_num,[[mono_params_gore.loc[spes[0],'r'],mono_params_gore.loc[spes[1],'r']],[interacts_gore.loc[spes[0],spes[1]],interacts_gore.loc[spes[1],spes[0]]]],x0,endt)
    # realiz_ax[1].set_title('Parameters from Friedman, Higgens, & Gore (2017)')
    # realiz_ax[1].step(stoch2[0],stoch2[1][:,0],'r',label = spes[0])
    # realiz_ax[1].step(stoch2[0],stoch2[1][:,1],'b',label = spes[1])
    # realiz_ax[1].plot(times2,sols2[:,0],'r:',times2,sols2[:,1], 'b:')
    # realiz_ax[1].legend()
    svnm = '../stochastic_realizations/'+'_'.join(spes)+'.jpg'
    realiz_fig.savefig(svnm)

#for the trio model
for spes in real_outs.index:
    rxnrates = make_k_trio(interact_params,mono_params_df['r'],spes)
    x0 = rand(3)
    endt = 20
    stoch = gillep(three_spec_sources,three_spec_prodcts,rxnrates,x0,endt,N = 100)
    sols,times  = solve_vode(tthreespec_num,[[mono_params_df.loc[spes[0],'r'],mono_params_df.loc[spes[1],'r'],mono_params_df.loc[spes[2],'r']],[interact_params.loc[spes[0],spes[1]],interact_params.loc[spes[0],spes[2]],interact_params.loc[spes[1],spes[0]],interact_params.loc[spes[1],spes[2]],interact_params.loc[spes[2],spes[0]],interact_params.loc[spes[2],spes[1]]]],x0,endt)
    realiz_fig,realiz_ax = plt.subplots(1,1,figsize = (15,5))
    realiz_fig.suptitle(spes[0]+', '+spes[1]+' & '+spes[2],fontsize = 20)
    realiz_ax.set_title('Pair Fitted Parameters')
    realiz_ax.set_xlabel('time')
    realiz_ax.set_ylabel('population size')
    realiz_ax.step(stoch[0],stoch[1][:,0],'r',label = spes[0])
    realiz_ax.step(stoch[0],stoch[1][:,1],'b',label = spes[1])
    realiz_ax.step(stoch[0],stoch[1][:,2],'g',label = spes[2])
    realiz_ax.plot(times,sols[:,0],'r:',times,sols[:,1], 'b:',times,sols[:,2],':g')
    realiz_ax.legend(fontsize = 20)
    # rxnrates2 = make_k_trio(interacts_gore,mono_params_gore['r'],spes)
    # stoch2 = gillep(three_spec_sources,three_spec_prodcts,rxnrates2,x0,endt,N = 100)
    # sols2,times2  = solve_vode(tthreespec_num,[[mono_params_gore.loc[spes[0],'r'],mono_params_gore.loc[spes[1],'r'],mono_params_gore.loc[spes[2],'r']],[interacts_gore.loc[spes[0],spes[1]],interacts_gore.loc[spes[0],spes[2]],interacts_gore.loc[spes[1],spes[0]],interacts_gore.loc[spes[1],spes[2]],interacts_gore.loc[spes[2],spes[0]],interacts_gore.loc[spes[2],spes[1]]]],x0,endt)
    # realiz_ax[1].set_title('Parameters from Friedman, Higgens, & Gore (2017)')
    # realiz_ax[1].step(stoch2[0],stoch2[1][:,0],'r',label = spes[0])
    # realiz_ax[1].step(stoch2[0],stoch2[1][:,1],'b',label = spes[1])
    # realiz_ax[1].step(stoch2[0],stoch2[1][:,2],'g',label = spes[2])
    # realiz_ax[1].plot(times2,sols2[:,0],'r:',times2,sols2[:,1], 'b:',times2,sols2[:,2],':g')
    # realiz_ax[1].legend()
    svnm = '../stochastic_realizations/'+'_'.join(spes)+'.jpg'
    realiz_fig.savefig(svnm)

####Now the interesting part: estimate the probability of each outcome.
volumes = [10,100,1000,10000]
#for each pair
ext_prob_pairs = {}
ext_prob_trios = {}
ext_prob_pairs_gore = {}
ext_prob_trios_gore = {}

pairs_conf_ints = {}
pairs_conf_ints_gore = {}
trios_conf_ints = {}
trios_conf_ints_gore = {}

monte_pair_out = pair_outs_fit.copy()
monte_pair_out_gore = pair_outs_gore.copy()
monte_trio_out = real_outs.copy()
monte_trio_out_gore = real_outs.copy()

t1 = time.time()
for N in volumes:
    stop_time = 20
    number_of_sims = 1000
    ext_prob_pairs[N] = {}
    ext_prob_pairs_gore[N] = {}
    pairs_conf_ints[N] = {}
    pairs_conf_ints_gore[N] = {}
    ################### HERE IS THE MONTE CARLO PAIR EXPERIMENT ################################
    for spes in pair_outs_fit.index:
        ext_prob_pairs[N][spes],pairs_conf_ints[N][spes] = extinction_prob_pair_tl(spes,interact_params,mono_params_df['r'],vol = N,maxsims =number_of_sims,endt = stop_time)
        ext_prob_pairs_gore[N][spes],pairs_conf_ints_gore[N][spes] = extinction_prob_pair_tl(spes,interacts_gore,mono_params_gore['r'],vol = N,maxsims =number_of_sims,endt = stop_time)
    #then add to the outcome DataFrame
    pair_outcome_probabilities = pd.Series(index = pair_outs_fit.index)
    pair_likely_outcome = pd.Series(index= pair_outs_fit.index, dtype = 'object')
    for pr in pair_outs_fit.index:
        pair_outcome_probabilities[[pr]]  = ext_prob_pairs[N][pr][pair_outs_fit.loc[[pr],'Observed'][0]]
        max_prob = max(ext_prob_pairs[N][pr].values())
        pair_likely_outcome[[pr]]= [[(key,max_prob) for key in ext_prob_pairs[N][pr] if ext_prob_pairs[N][pr][key]==max_prob]]
    monte_pair_out['Probability of Observation: N='+str(N)] = pair_outcome_probabilities
    monte_pair_out['Most Likely Observation: N='+str(N)] = pair_likely_outcome
    ##Gore
    pair_outcome_probabilities_gore = pd.Series(index = pair_outs_fit.index)
    pair_likely_outcome_gore = pd.Series(index= pair_outs_fit.index, dtype = 'object')
    for pr in pair_outs_fit.index:
        pair_outcome_probabilities_gore[[pr]]  = ext_prob_pairs_gore[N][pr][pair_outs_fit.loc[[pr],'Observed'][0]]
        max_prob = max(ext_prob_pairs_gore[N][pr].values())
        pair_likely_outcome_gore[[pr]]= [[(key,max_prob) for key in ext_prob_pairs_gore[N][pr] if ext_prob_pairs_gore[N][pr][key]==max_prob]]
    monte_pair_out_gore['Probability of Observation: N='+str(N)] = pair_outcome_probabilities_gore
    monte_pair_out_gore['Most Likely Observation: N='+str(N)] = pair_likely_outcome_gore
    ####################Trios:
    ext_prob_trios[N] = {}
    ext_prob_trios_gore[N] = {}
    trios_conf_ints[N] = {}
    trios_conf_ints_gore[N] = {}
    ################### HERE IS THE MONTE CARLO TRIO EXPERIMENT ################################
    for spes in real_outs.index:
        ext_prob_trios[N][spes],trios_conf_ints[N][spes] = extinction_prob_trio_tl(spes,interact_params,mono_params_df['r'],vol = N,maxsims =number_of_sims,endt = stop_time)
        ext_prob_trios_gore[N][spes],trios_conf_ints_gore[N][spes] = extinction_prob_trio_tl(spes,interacts_gore,mono_params_df['r'],vol = N,maxsims =number_of_sims,endt = stop_time)
    #then add to the outcome DataFrame
    trio_outcome_probabilities = pd.Series(index = real_outs.index)
    trio_likely_outcome = pd.Series(index= real_outs.index, dtype = 'object')
    for tri in real_outs.index:
        trio_outcome_probabilities[[tri]]  = ext_prob_trios[N][tri][real_outs.loc[[tri],'Observed'][0]]
        max_prob = max(ext_prob_trios[N][tri].values())
        trio_likely_outcome[[tri]]= [[(key,max_prob) for key in ext_prob_trios[N][tri] if ext_prob_trios[N][tri][key]==max_prob]]
    monte_trio_out['Probability of Observation: N='+str(N)] = trio_outcome_probabilities
    monte_trio_out['Most Likely Observation: N=' + str(N)] = trio_likely_outcome
    ####Gore
    trio_outcome_probabilities_gore = pd.Series(index = real_outs.index)
    trio_likely_outcome_gore = pd.Series(index= real_outs.index, dtype = 'object')
    for tri in real_outs.index:
        trio_outcome_probabilities_gore[[tri]]  = ext_prob_trios_gore[N][tri][real_outs.loc[[tri],'Observed'][0]]
        max_prob = max(ext_prob_trios_gore[N][tri].values())
        trio_likely_outcome_gore[[tri]]= [[(key,max_prob) for key in ext_prob_trios_gore[N][tri] if ext_prob_trios_gore[N][tri][key]==max_prob]]
    monte_trio_out_gore['Probability of Observation: N='+str(N)] = trio_outcome_probabilities_gore
    monte_trio_out_gore['Most Likely Observation: N=' + str(N)] = trio_likely_outcome_gore
    t2 = time.time() - t1
    print(t2/60)





monte_pair_out.to_pickle('monte_pair_out')
monte_pair_out_gore.to_pickle('monte_pair_out_gore')
monte_trio_out.to_pickle('monte_trio_out')
monte_trio_out_gore.to_pickle('monte_trio_out_gore')

hminds = monte_pair_out.columns.str.contains('Probability of Observation')
import seaborn as sns


pair_labels = [sp[0] + ', ' + sp[1] for sp in monte_pair_out.index]
trio_labels = [sp[0] + ', ' + sp[1] + ', ' + sp[2] for sp in monte_trio_out.index]

monte_pair_out.index = pair_labels
monte_trio_out.index = trio_labels

monte_pair_out.loc[:,hminds].rename(dict(zip(monte_pair_out.loc[:,hminds].columns,monte_pair_out.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1).T

hmfig = plt.subplots(1,1,figsize = (15,3))
sns.heatmap(monte_pair_out.loc[:,hminds].rename(dict(zip(monte_pair_out.loc[:,hminds].columns,monte_pair_out.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1).T, ax = hmfig[1],cmap = 'YlGnBu')
hmfig[0].tight_layout()
hmfig[0].savefig('../stochastic_heatmaps/pairs_est_transp')

hmfig_an = plt.subplots(1,1,figsize = (5,15))
sns.heatmap(monte_pair_out.loc[:,hminds].rename(dict(zip(monte_pair_out.loc[:,hminds].columns,monte_pair_out.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1), ax = hmfig_an[1],cmap = 'YlGnBu', annot = True)
hmfig_an[0].tight_layout()
hmfig_an[0].savefig('../stochastic_heatmaps/pairs_est_annot')

hmfig2 = plt.subplots(1,1,figsize = (5,15))
sns.heatmap(monte_pair_out_gore.loc[:,hminds].rename(dict(zip(monte_pair_out_gore.loc[:,hminds].columns,monte_pair_out_gore.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1), ax = hmfig2[1],cmap = 'YlGnBu')
hmfig2[0].tight_layout()
hmfig2[0].savefig('../stochastic_heatmaps/pairs_gore')

hmfig2_an = plt.subplots(1,1,figsize = (5,15))
sns.heatmap(monte_pair_out_gore.loc[:,hminds].rename(dict(zip(monte_pair_out_gore.loc[:,hminds].columns,monte_pair_out_gore.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1), ax = hmfig2_an[1],cmap = 'YlGnBu',annot = True)
hmfig2_an[0].tight_layout()
hmfig2_an[0].savefig('../stochastic_heatmaps/pairs_gore_annot')

hmfig3 = plt.subplots(1,1,figsize = (15,3))
sns.heatmap(monte_trio_out.loc[:,hminds].rename(dict(zip(monte_trio_out.loc[:,hminds].columns,monte_trio_out.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1).T, ax = hmfig3[1],cmap = 'YlGnBu')
hmfig3[0].tight_layout()
hmfig3[0].savefig('../stochastic_heatmaps/trios_est_transp')

hmfig3_an = plt.subplots(1,1,figsize = (5,15))
sns.heatmap(monte_trio_out.loc[:,hminds].rename(dict(zip(monte_trio_out.loc[:,hminds].columns,monte_trio_out.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1), ax = hmfig3_an[1],cmap = 'YlGnBu', annot = True)
hmfig3_an[0].tight_layout()
hmfig3_an[0].savefig('../stochastic_heatmaps/trios_est_annot')

hmfig4 = plt.subplots(1,1,figsize = (5,15))
sns.heatmap(monte_trio_out_gore.loc[:,hminds].rename(dict(zip(monte_trio_out_gore.loc[:,hminds].columns,monte_trio_out_gore.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1), ax = hmfig4[1],cmap = 'YlGnBu')
hmfig4[0].tight_layout()
hmfig4[0].savefig('../stochastic_heatmaps/trios_gore')

hmfig4_an = plt.subplots(1,1,figsize = (5,15))
sns.heatmap(monte_trio_out_gore.loc[:,hminds].rename(dict(zip(monte_trio_out_gore.loc[:,hminds].columns,monte_trio_out_gore.loc[:,hminds].columns.str.replace('Probability of Observation: ',''))),axis = 1), ax = hmfig4_an[1],cmap = 'YlGnBu', annot = True)
hmfig4_an[0].tight_layout()
hmfig4_an[0].savefig('../stochastic_heatmaps/trios_gore_annot')
