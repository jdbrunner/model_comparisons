#script to load the outcomes of the gore experiments, as well as the more detailed data from them
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

#For testing, run the module this way so you can update it.
#%run lv_pair_trio_functs.py

#Running the module also loads:
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
#so one can comment this out if running this script independently

########################### List of outcomes of triplet experiment
indx = ['Ea','Pa','Pch','Pci','Pf','Pp','Pv','Sm']
species_names = {'Ea':'Enterobacter aerogenes', 'Pa':'Pseudomonas aurantiaca', 'Pch':'Pseudomonas chlororaphis','Pci':'Pseudomonas citronellolis', 'Pf':'Pseudomonas fluorescens','Pp':'Pseudomonas putida', 'Pv':'Pseudomonas veronii', 'Sm':'Serratia marcescens'}
# Read and process the set of measured triplet outcomes from the paper.
real_outs = pd.read_csv('gore_data/trio_outcomes.csv')
real_outs.columns = list('ABCDEF')

real_outs.drop(0,inplace = True)
real_outs.loc[29,['D','E','F']] = real_outs.loc[28,['D','E','F']]
real_outs.drop(28,inplace = True)
real_outs.drop(30,inplace = True)
real_outs.drop(31,inplace = True)
real_outs.drop(32,inplace = True)
rtris = []

for i in real_outs.index:
    rtris = rtris + [(real_outs.loc[i,'A'],real_outs.loc[i,'B'],real_outs.loc[i,'C'])]

real_outs.index = rtris
real_outs.drop(['A','B','C'], axis = 1, inplace = True)
real_outs.columns = ['First','Second','Third']

outcome = pd.Series(index = real_outs.index)


for i in outcome.index:
    outcome.loc[i] = '-'.join(list(np.array(i)[np.where(*real_outs.loc[[i]].values.astype('int').astype('bool'))]))

real_outs.loc[:,'Observed'] = outcome
real_outs.drop(['First','Second','Third'], axis = 1, inplace = True)


pairs = list(it.combinations(indx,2))

pair_outs_gore = pd.DataFrame(index = list(pairs))

pair_outs_gore_li = ['Ea-Pa','Ea-Pch','Ea-Pci','Ea-Pf','Pp','Pv','Ea','Pch','Pa','Pa-Pf','Pa-Pp','Pa-Pv','Pa-Sm','Pch','Pch','Pch-Pp','Pch-Pv','Pch-Sm','Pci-Pf','Pp','Pv','Pci-Sm','Pf-Pp','Pf-Pv','Pf-Sm','Pp','Pp','Pv-Sm']
pair_outs_gore.loc[:,'Observed']= pair_outs_gore_li



pair_outs_gore.loc[[('Ea','Pp')],'Observed'] = 'Ea-Pp'
pair_outs_gore.loc[[('Ea','Pv')],'Observed'] = 'Ea-Pv'

##################### Parameters reported in the supplimental of the Gore paper
mono_params_gore = pd.DataFrame(index = indx, columns = ['r','K'])

mono_params_gore['r'] = mono_params_gore.index.map({'Ea':0.46, 'Pa':0.55,'Pch':0.18,'Pci':0.16,'Pf':0.25,'Pp':0.65,'Pv':0.57,'Sm':0.34})
mono_params_gore['K'] = mono_params_gore.index.map({'Ea':0.13, 'Pa':0.07,'Pch':0.11,'Pci':0.01,'Pf':0.05,'Pp':0.14,'Pv':0.11,'Sm':0.15})

inter_rep = np.array([[1,0.69,1.09,0.55,1.53,0.82,1.09,0.72],[-0.18,1,2.44,-2.58,1.13,0.43,0.01,0.21],[-0.11,-0.8,1,-15.75,0.29,-0.04,-0.05,-0.03],[-0.32,0,0.18,1,-3.39,0,0.05,-0.3],[-0.02,0.28,1.2,0.83,1,0.01,0.07,-0.1],[0.87,1.58,1.24,0.24,1,1,1.01,0.84],[0.83,0.28,0.47,0,-0.02,0.79,1,0.7],[0.96,1.23,1.42,1.21,1.31,0.91,0.98,1]])
interacts_gore_unscld = pd.DataFrame(inter_rep,index = indx,columns = indx)
interacts_gore = rescale_model(interacts_gore_unscld,mono_params_gore['K'])

##################### Fix the Gore parameters to match the pair outcomes
pair_out_gore = test_params_LV(interacts_gore, mono_params_gore['K'], mono_params_gore['r'], real_outs, pair_outs_gore)[1]

interacts_gore_pf = interacts_gore.copy()
#Go through and adjust parameters so that the stability of the pair matches observation
for ind in pair_out_gore[pair_out_gore.LVRight == False].index:
    pre_only = ''.join(set(pair_out_gore.loc[[ind],'LVPrediction'][0].split('-'))-set(pair_out_gore.loc[[ind],'Observed'][0].split('-')))#survivor only in predicted
    obs_only = ''.join(set(pair_out_gore.loc[[ind],'Observed'][0].split('-'))-set(pair_out_gore.loc[[ind],'LVPrediction'][0].split('-')))#survivor only in observed
    if len(pre_only):#if you're in predicted but not observed, let's kill you
        othe = list(set(ind) - {pre_only})[0]
        interacts_gore_pf.loc[pre_only,othe] = -1.001
    if len(obs_only):#if you're in observed but not prediced, let's save you!
        othe = list(set(ind) - {obs_only})[0]
        interacts_gore_pf.loc[obs_only,othe] = -0.999


##################### Timecourse Data
#### Import the data from excel - each become a dict of DFs, which are multi-indexed
#### with experiment number and time point number.
mono_data = pd.read_excel('gore_data/monoculture_timeSeries.xlsx',sheet_name = None)
pair_data = pd.read_excel('gore_data/pair_timeSeries.xlsx',sheet_name = None)


for prdf in pair_data:
    pair_data[prdf].dropna(inplace = True)
    whexps = list(np.where(pair_data[prdf].index.get_loc(0))[0]) + [len(pair_data[prdf].index)]
    explist = list(it.chain(*[[j]*(whexps[j]-whexps[j-1]) for j in range(1,len(whexps))]))
    pair_data[prdf].set_index([explist,pair_data[prdf].index], inplace = True)
    pair_data[prdf].index.rename(['Experiment','Time'],inplace = True)
trio_data = pd.read_excel('gore_data/trio_lastTransfer.xlsx',sheet_name = None)
for trdf in trio_data:
    trio_data[trdf].dropna(inplace = True)
    whexps = list(np.where(trio_data[trdf].index.get_loc(0))[0]) + [len(trio_data[trdf].index)]
    explist = list(it.chain(*[[j]*(whexps[j]-whexps[j-1]) for j in range(1,len(whexps))]))
    trio_data[trdf].set_index([explist,trio_data[trdf].index], inplace = True)
    trio_data[trdf].index.rename(['Experiment','Time'],inplace = True)
groups_data = pd.read_excel('gore_data/7and8Species_lastTransfer.xlsx',sheet_name = None)
for grpdf in groups_data:
    groups_data[grpdf].dropna(inplace = True)
    whexps = list(np.where(groups_data[grpdf].index.get_loc(0))[0]) + [len(groups_data[grpdf].index)]
    explist = list(it.chain(*[[j]*(whexps[j]-whexps[j-1]) for j in range(1,len(whexps))]))
    groups_data[grpdf].set_index([explist,groups_data[grpdf].index], inplace = True)
    groups_data[grpdf].index.rename(['Experiment','Time'],inplace = True)
