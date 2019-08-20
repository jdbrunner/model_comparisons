
from lv_pair_trio_functs import *
#Running the module also loads:
# import numpy as np
# import itertools as it
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
# from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
# from scipy.integrate import ode
# from numpy.random import rand

from load_gore_results import *
#Creates a DataFrame of the observed outcomes of the trio experiments (real_outs)
#Creates a DataFrame of the observed outcomes of the pair experiment "by the eye test" (pair_outs_gore)
#Creates a DataFrame of the single species timecourse experiments (mono_data)
#Creates a DataFrame of the pair experiment timecourse data (pair_data)
#Creates a DataFrame of the trio experiment (only start/end) (trio_data)
#Creates a DataFrame of the group experiment data (only start/end) (groups_data)
#Creates a Dict of species names (species_names)


#we have some survival if we choose all our kj1, dj in [1,2].
f1 = 1
dst = 0.49

#first - any big winners? Let's start with those.

scores = {}
for i in indx:
    scores[i] = 0
for ky in pair_outs_gore.index:
    if '-' not in pair_outs_gore.loc[[ky],'Observed'].values[0]:
        lser = np.array(ky)[[s not in pair_outs_gore.loc[[ky],'Observed'].values[0] for s in ky]][0]
        scores[pair_outs_gore.loc[[ky],'Observed'].values[0]] += 1
        scores[lser] -= 1


srtscores = sorted(scores.items(), key=lambda kv: kv[1], reverse = True)
winorder = [k[0] for k in srtscores]


k1opts = np.arange(1.1,1.9,0.1)[::-1]

k1vals = dict(zip(winorder,k1opts))
dvals = dict(zip(winorder,np.ones(len(winorder))))


#### Function to return the vector field for the pair model so we can run the thing
def run_pair_mod(t,y,pars):
    [x1,x2,y1,y2] = y
    [k11,k21,d1,d2,a,b,k12,k22] = pars
    # k11 = k1vals[pr[0]]
    # k21 = k1vals[pr[1]]
    # d1 = dvals[pr[0]]
    # d2 = dvals[pr[1]]
    # k12 = pair_models.loc[[pr],'k12'].values[0]
    # k22 = pair_models.loc[[pr],'k22'].values[0]
    # a = pair_models.loc[[pr],'a'].values[0]
    # b = pair_models.loc[[pr],'b'].values[0]
    dx1dt = k11*x1*y1 - d1*x1 + k12*x1*y2
    dx2dt = k21*x2*y1 - d2*x2 + k22*x2*y2
    dy1dt = f1 - dst*y1 -  k11*x1*y1 - k21*x2*y1
    dy2dt = a*k11*x1*y1 + b*k21*x2*y1 - dst*y2 - k12*x1*y2 - k22*x2*y2
    return [dx1dt,dx2dt,dy1dt,dy2dt]

def run_trio_mod_unadj(t,y,parms):
    [x1,x2,x3,y1,y2,y3,y4] = y
    [d1,d2,d3,k11,k21,k31,a1,a2,a3,b1,b2,b3,k12,k22,k33,k13,k24,k34] = parms# list(trios_unadjusted.loc[[tr]].values[0])
    dx1dt = k11*x1*y1 - d1*x1 + k12*x1*y2 + k13*x1*y3
    dx2dt = k21*x2*y1 - d2*x2 + k22*x2*y2 + k24*x2*y4
    dx3dt = k31*x3*y1 - d3*x3 + k33*x3*y3 + k34*x3*y4
    dy1dt = f1 - dst*y1 -  k11*x1*y1 - k21*x2*y1 - k31*x3*y1
    dy2dt = a1*k11*x1*y1 + b1*k21*x2*y1 - dst*y2 - k12*x1*y2 - k22*x2*y2
    dy3dt = a2*k11*x1*y1 + b2*k31*x3*y1 - dst*y3 - k13*x1*y3 - k33*x3*y3
    dy4dt = a3*k21*x2*y1 + b3*k31*x3*y1 - dst*y4 - k24*x2*y4 - k34*x3*y4
    return [dx1dt,dx2dt,dx3dt,dy1dt,dy2dt,dy3dt,dy4dt]






def run_trio_mod(t,y,parms):
    [x1,x2,x3,y1,y2,y3,y4,y5] = y
    [d1,d2,d3,k11,k21,k31,a1,a2,a3,b1,b2,b3,k12,k22,k33,k13,k24,k34,c1,c2,c3,psi15,psi25,psi35] = parms# list(trios_unadjusted.loc[[tr]].values[0])
    dx1dt = k11*x1*y1 - d1*x1 + k12*x1*y2 + k13*x1*y3 + psi15*x1*y5
    dx2dt = k21*x2*y1 - d2*x2 + k22*x2*y2 + k24*x2*y4 + psi25*x2*y5
    dx3dt = k31*x3*y1 - d3*x3 + k33*x3*y3 + k34*x3*y4 + psi35*x3*y5
    dy1dt = f1 - dst*y1 -  k11*x1*y1 - k21*x2*y1
    dy2dt = a1*k11*x1*y1 + b1*k21*x2*y1 - dst*y2 - k12*x1*y2 - k22*x2*y2
    dy3dt = a2*k11*x1*y1 + b2*k31*x3*y1 - dst*y3 - k13*x1*y3 - k33*x3*y3
    dy4dt = a3*k21*x2*y1 + b3*k31*x3*y1 - dst*y4 - k24*x2*y4 - k34*x3*y4
    dy5dt = c3*k12*x1*y2 + c2*k13*x1*y3 + c3*k22*x2*y2 + c1*k24*x2*y4 + c2*k33*x3*y3 + c1*k34*x3*y4 - 0.1*dst*y5 - psi15*x1*y5 - psi25*x2*y5 - psi35*x3*y5
    return [dx1dt,dx2dt,dx3dt,dy1dt,dy2dt,dy3dt,dy4dt,dy5dt]

#now add whatever cross-feeding or cross poisoning we need.
##Start by figuring out current outcomes



def count_it(pair_outcome_df,trio_outcome_df, k1vs, dvs):

    ##### Figure out what pathways we need for pairs
    winner_nocross = pd.Series(index = pair_outcome_df.index)
    for pr in pair_outcome_df.index:
        winner = pr[np.argmax([k1vs[pr[0]]/dvs[pr[0]],k1vs[pr[1]]/dvs[pr[1]]])]
        winner_nocross[pr] = winner


    pair_task = pair_outcome_df.copy()
    pair_task['NoXtalk'] = winner_nocross
    # pair_task
    needs_p = pd.Series(index = pair_task.index)
    for pr in pair_task.index:
        obse = pair_task.loc[[pr],'Observed'].values[0]
        nox = pair_task.loc[[pr],'NoXtalk'].values[0]
        if  obse == nox:
            needs_p[pr] = 'None' #needs nothing
        elif (pr[0] in nox) and ('-' in obse):
            needs_p[pr] = pr[0] + '_XF_' + pr[1] #needs cross feeding
        elif (pr[1] in nox) and ('-' in obse):
            needs_p[pr] = pr[1] + '_XF_' + pr[0] #needs cross feeding
        elif pr[0] == nox and pr[1] == obse:
            needs_p[pr] = pr[1] + '_XP_' + pr[0] #needs cross poisoning
        else:
            needs_p[pr] = pr[0] + '_XP_' + pr[1] #needs cross poisoning


    pair_task['Needs'] = needs_p


    pair_models = pd.DataFrame(index = pair_task.index)
    acol = pd.Series(index = pair_models.index)
    bcol = pd.Series(index = pair_models.index)
    k12col = pd.Series(index = pair_models.index)
    k22col = pd.Series(index = pair_models.index)
    #, columns = ['a','b','k12','k22'])
    for pr in pair_models.index:
        d1 = dvals[pr[0]]
        d2 = dvals[pr[1]]
        k11 = k1vals[pr[0]]
        k21 = k1vals[pr[1]]
        if pair_task.loc[[pr],'Needs'].values[0] == pr[0] + '_XF_' + pr[1]: #needs cross feeding
            acol[pr] =1
            bcol[pr]  = 0
            k12col[pr]  = 0
            k22col[pr] =  1.5*(1/(f1-(dst*d1)/k11))*(d2 - k21*d1/k11)
        elif pair_task.loc[[pr],'Needs'].values[0] == pr[1] + '_XF_' + pr[0]:
            acol[pr] =0
            bcol[pr]  = 1
            k12col[pr]  = 1.5*(1/(f1-(dst*d2)/k21))*(d1 - k11*d2/k21)
            k22col[pr] = 0
        elif pair_task.loc[[pr],'Needs'].values[0] == pr[0] + '_XP_' + pr[1]:
            acol[pr] =1
            bcol[pr]  = 0
            k12col[pr]  = 0
            k22col[pr] =  1.5*(1/(f1-(dst*d1)/k11))*(d2 - k21*d1/k11)
        elif pair_task.loc[[pr],'Needs'].values[0] == pr[1] + '_XP_' + pr[0]:
            acol[pr] =0
            bcol[pr]  = 1
            k12col[pr]  = 1.5*(1/(f1-(dst*d2)/k21))*(d1 - k11*d2/k21)
            k22col[pr] = 0
        else:
            acol[pr] = 0
            bcol[pr]  = 0
            k12col[pr]  =  0
            k22col[pr] = 0


    pair_models['a'] = acol
    pair_models['b'] = bcol
    pair_models['k12'] = k12col
    pair_models['k22'] = k22col


    trio_model_outcomes = trio_outcome_df.copy()
    trios_unadjusted = pd.DataFrame(index = trio_outcome_df.index)
    a1col = pd.Series(index = trio_outcome_df.index)
    a2col = pd.Series(index = trio_outcome_df.index)
    a3col = pd.Series(index = trio_outcome_df.index)
    b1col = pd.Series(index = trio_outcome_df.index)
    b2col = pd.Series(index = trio_outcome_df.index)
    b3col = pd.Series(index = trio_outcome_df.index)

    k11col = pd.Series(index = trio_outcome_df.index)
    k21col = pd.Series(index = trio_outcome_df.index)
    k31col = pd.Series(index = trio_outcome_df.index)

    d1col = pd.Series(index = trio_outcome_df.index)
    d2col = pd.Series(index = trio_outcome_df.index)
    d3col = pd.Series(index = trio_outcome_df.index)

    k12col = pd.Series(index = trio_outcome_df.index)
    k22col = pd.Series(index = trio_outcome_df.index)
    k33col = pd.Series(index = trio_outcome_df.index)
    k13col = pd.Series(index = trio_outcome_df.index)
    k24col = pd.Series(index = trio_outcome_df.index)
    k34col = pd.Series(index = trio_outcome_df.index)

    for tr in trios_unadjusted.index:
        k11col[tr] = k1vals[tr[0]]
        k21col[tr] = k1vals[tr[1]]
        k31col[tr] = k1vals[tr[2]]
        d1col[tr] = dvals[tr[0]]
        d2col[tr] = dvals[tr[1]]
        d3col[tr] = dvals[tr[2]]

        ab = (tr[0],tr[1])
        if ab in pair_models.index:
            a1col[tr] = pair_models.loc[[ab],'a']
            b1col[tr] = pair_models.loc[[ab],'b']
            k12col[tr] = pair_models.loc[[ab],'k12']
            k22col[tr] = pair_models.loc[[ab],'k22']
        else:
            ab = (tr[1],tr[0])
            a1col[tr] = pair_models.loc[[ab],'b']
            b1col[tr] = pair_models.loc[[ab],'a']
            k12col[tr]= pair_models.loc[[ab],'k22']
            k22col[tr] = pair_models.loc[[ab],'k12']

        ac = (tr[0],tr[2])
        if ac in pair_models.index:
            a2col[tr] = pair_models.loc[[ac],'a']
            b2col[tr] = pair_models.loc[[ac],'b']
            k13col[tr] = pair_models.loc[[ac],'k12']
            k33col[tr] = pair_models.loc[[ac],'k22']
        else:
            ac = (tr[2],tr[0])
            a2col[tr] = pair_models.loc[[ac],'b']
            b2col[tr] = pair_models.loc[[ac],'a']
            k13col[tr] = pair_models.loc[[ac],'k22']
            k33col[tr] = pair_models.loc[[ac],'k12']

        bc = (tr[1],tr[2])
        if bc in pair_models.index:
            a3col[tr] = pair_models.loc[[bc],'a']
            b3col[tr] = pair_models.loc[[bc],'b']
            k24col[tr] = pair_models.loc[[bc],'k12']
            k34col[tr] = pair_models.loc[[bc],'k22']
        else:
            bc = (tr[2],tr[1])
            a3col[tr] = pair_models.loc[[bc],'b']
            b3col[tr] = pair_models.loc[[bc],'a']
            k24col[tr] = pair_models.loc[[bc],'k22']
            k34col[tr] = pair_models.loc[[bc],'k12']

    trios_unadjusted['d1'] = d1col
    trios_unadjusted['d2'] = d2col
    trios_unadjusted['d3'] = d3col

    trios_unadjusted['k11'] = k11col
    trios_unadjusted['k21'] = k21col
    trios_unadjusted['k31'] = k31col

    trios_unadjusted['a1'] = a1col
    trios_unadjusted['a2'] = a2col
    trios_unadjusted['a3'] = a3col
    trios_unadjusted['b1'] = b1col
    trios_unadjusted['b2'] = b2col
    trios_unadjusted['b3'] = b3col

    trios_unadjusted['k12'] = k12col
    trios_unadjusted['k22'] = k22col
    trios_unadjusted['k33'] = k33col
    trios_unadjusted['k13'] = k13col
    trios_unadjusted['k24'] = k24col
    trios_unadjusted['k34'] = k34col






    trio_model_outcomes_unadj = pd.Series(index = trio_outcome_df.index)

    for tr in trio_outcome_df.index:
        tr_mod = ode(run_trio_mod_unadj)
        tr_mod.set_initial_value([1,1,1,1,0,0,0],0).set_f_params(list(trios_unadjusted.loc[[tr]].values[0]))
        while tr_mod.successful() and tr_mod.t < 500:
            result = tr_mod.integrate(tr_mod.t + 1)
        trio_model_outcomes_unadj[tr] = np.array(tr)[result[:3].round(8).astype('bool')]





    compare_pt = pd.DataFrame(index = trio_outcome_df.index, columns = ['PairPaths','Trio_Res','PP_Res','ABS_Change','CanFix'])
    compare_pt.Trio_Res = trio_outcome_df.Observed
    compare_pt.PP_Res = trio_model_outcomes_unadj
    # compare_pt.TrioPaths = trio_task.Needs
    ppaths = pd.Series(index = compare_pt.index, dtype = object)

    for tr in compare_pt.index:
        ptli = []
        for pr in pair_task.index:
            if all([s in tr for s in pr]):
                if pair_task.loc[[pr],'Needs'].values[0] != 'None':
                    ptli += [pair_task.loc[[pr],'Needs'].values[0]]
        ppaths[tr] = ptli

    compare_pt.PairPaths = ppaths

    abs_chng  = pd.Series(index = trio_outcome_df.index,dtype = object)
    canfx = pd.Series(index = trio_outcome_df.index)
    for tr in compare_pt.index:
        ch = [compare_pt.Trio_Res[tr] + '_XP_' + sp for sp in compare_pt.PP_Res[tr] if sp not in compare_pt.Trio_Res[tr].split('-')] + ['-'.join(compare_pt.PP_Res[tr]) + '_XF_' + sp for sp in compare_pt.Trio_Res[tr].split('-') if sp not in compare_pt.PP_Res[tr]]
        abs_chng[tr] = ch
        if all([len(p.split('-')) == 2 for p in ch]):
            canfx[tr] = all([any([all([sps in pway for sps in pt.split('_')[0].split('-')]) for pway in compare_pt.PairPaths[tr]]) for pt in ch])
        else:
            canfx[tr] = False


    compare_pt.ABS_Change = abs_chng
    compare_pt.CanFix = canfx

    all_by_prods = list(k1vs.keys()) + [pth for pth in pair_task.Needs if pth != 'None']

    all_x_targets = pd.Series([[]]*len(all_by_prods),index = all_by_prods, dtype = object)


    for pth in [pth for pth in pair_task.Needs if pth != 'None']:
        pthspl = pth.split('_')
        all_x_targets.loc[pthspl[0]] = all_x_targets.loc[pthspl[0]] + [pthspl[1]+'_'+pthspl[2]]

    for tr in [tro for tro in compare_pt.index if compare_pt.CanFix[tro]]:
        for pthneeded in compare_pt.ABS_Change[tr]:
            spl1 = pthneeded.split('_')
            spl2 = spl1[0].split('-')
            ##What it is the 2ary path
            pth = [pway for pway in compare_pt.PairPaths[tr] if all([sps in pway for sps in spl2])][0]
            spl3 = pth.split('_')
            all_x_targets.loc[spl3[0]] = all_x_targets.loc[spl3[0]] + [spl3[1]+'-'+spl3[2]+'_'+spl1[1]+'_'+spl1[2]]
            all_x_targets.loc[pth] = all_x_targets.loc[pth] + [spl1[1]+'_'+spl1[2]]



    return compare_pt,all_x_targets



def minitmaybe(pair_df,trio_df,dvals, order):
    k1opts = np.arange(1.1,1.9,0.1)[::-1]
    k1vals = dict(zip(order,k1opts))
    score_dat = count_it(pair_df,trio_df,k1vals,dvals)
    num_u = len(np.unique([set(li) for li in score_dat[1].values if li != []]))
    perc_cov = sum(score_dat[0].CanFix)/(len(score_dat[0]))
    score = len(np.unique([set(li) for li in score_dat[1].values if li != []]))*len(score_dat[0])/sum(score_dat[0].CanFix)
    for i in range(200):
        new_ord = order.copy()
        j = np.random.randint(len(order))
        k = np.random.randint(len(order))
        new_ord[j] = order[k]
        new_ord[k] = order[j]
        new_k1s = dict(zip(new_ord,k1opts))
        new_score_dat = count_it(pair_df,trio_df,new_k1s,dvals)
        new_num_u = len(np.unique([set(li) for li in new_score_dat[1].values if li != []]))
        new_perc_cov = sum(new_score_dat[0].CanFix)/(len(new_score_dat[0]))
        new_score = new_num_u+1000*(1/new_perc_cov)
        if new_score < score:
            score = new_score
            order = new_ord
            num_u = new_num_u
            perc_cov = new_perc_cov
    return num_u,perc_cov,score,order


realdat = False
if realdat:
    do_it = minitmaybe(pair_outs_gore,real_outs,dvals,winorder)


    k1vals = dict(zip(do_it[1],k1opts))

    model = count_it(pair_outs_gore,real_outs,k1vals,dvals)

    plt_net = pd.DataFrame(columns =['Source','Target','Wght','Stype','Qual'])

    for x in do_it[1]:
        plt_net = plt_net.append(pd.DataFrame([['y1',x,1.0,'R1',1.0]],columns = plt_net.columns),ignore_index = True)
        plt_net = plt_net.append(pd.DataFrame([[x,'y1',-1.0,'R1',-1.0]],columns = plt_net.columns),ignore_index = True)

    j=2
    for y in np.unique([set(li) for li in model[1].values if li != []]):
        nm = 'y' + str(j)
        producers = list(model[1][[set(m1) == y for m1 in model[1]]].index)
        for sp in producers:
            spl = sp.split('_')[-1]
            plt_net = plt_net.append(pd.DataFrame([[spl,nm,1,'S',1]],columns = plt_net.columns),ignore_index = True)
        for tg in list(y):
            if '-' not in tg:
                tgs = tg.split('_')
                if tgs[0] == 'XF':
                    plt_net = plt_net.append(pd.DataFrame([[nm,tgs[1],1,'R2',1]],columns = plt_net.columns),ignore_index = True)
                    plt_net = plt_net.append(pd.DataFrame([[tgs[1],nm,-1,'S',-1]],columns = plt_net.columns),ignore_index = True)
                else:
                    plt_net = plt_net.append(pd.DataFrame([[nm,tgs[1],-1,'R2',-1]],columns = plt_net.columns),ignore_index = True)
                    plt_net = plt_net.append(pd.DataFrame([[tgs[1],nm,-1,'S',-1]],columns = plt_net.columns),ignore_index = True)
        j += 1


    plt_net.to_csv('min_met_network.csv')







def run_fake():
    pouts = pd.Series(index = pair_outs_gore.index)
    for pr in pouts.index:
        pouts[pr] = [pr[0] in pair_outs_gore['Observed'][pr], pr[1] in pair_outs_gore['Observed'][pr]]


    touts = pd.Series(index = real_outs.index)
    for tr in touts.index:
        touts[tr] = [tr[0] in real_outs['Observed'][tr], tr[1] in real_outs['Observed'][tr], tr[2] in real_outs['Observed'][tr]]

    some_pouts = pair_outs_gore.copy()
    aperm1 = np.random.permutation(pouts.values)
    k = 0
    for pr in some_pouts.index:
        some_pouts.loc[[pr],'Observed'] = '-'.join(np.array(pr)[aperm1[k]])
        k += 1

    some_touts = real_outs.copy()
    aperm2 = np.random.permutation(touts.values)
    k = 0
    for tr in some_touts.index:
        some_touts.loc[[tr],'Observed'] = '-'.join(np.array(tr)[aperm2[k]])
        k += 1


    rand_best = minitmaybe(some_pouts,some_touts,dvals,winorder)

    return np.array([rand_best[0],rand_best[1]])


from joblib import Parallel, delayed


def go_for(ti, addto = False):
    '''Generate models for random outcomes for "ti" minutes. It's going
    go over time by less than 12 min'''
    if addto:
        the_samples = list(np.load('outcome_samples.npy'))
    else:
        the_samples = []

    t1 = time.time()
    t = t1

    while t < t1 + 60*ti:#
        the_samples += Parallel(n_jobs=-2)(delayed(run_fake)()for i in range(10))###Each run takes ~10 min.
        t = time.time()
        print(len(the_samples))

    np.save('outcome_samples',the_samples)

    return

import sys
if len(sys.argv) > 1:
    timetodoit = float(sys.argv[1])
else:
    timetodoit = 10
# go_for(timetodoit, addto = True)

the_rands = list(np.load('outcome_samples.npy'))

import seaborn as sb



minmets = min(np.array(the_rands)[:,0])
maxmets = max(np.array(the_rands)[:,0])

mincov = min(np.array(the_rands)[:,1])


met_mu = np.mean(np.array(the_rands)[:,0])
met_sig = np.std(np.array(the_rands)[:,0])

print('Mean Metabolite Number: ',met_mu)
print('Std of Metabolite Number: ',met_sig)


mincov_mu = np.mean(np.array(the_rands)[:,1])
mincov_sig = np.std(np.array(the_rands)[:,1])

print('Mean Coverage: ',mincov_mu)
print('Std of Coverage: ',mincov_sig)

import scipy.stats as stats

x1 = np.linspace(met_mu - 3*met_sig, met_mu + 3*met_sig, 100)
x2 = np.linspace(mincov_mu - 3*mincov_sig, mincov_mu + 3*mincov_sig, 100)


fig, ax = plt.subplots(2,1,figsize=(15,10))
sb.distplot(np.array(the_rands)[:,0], ax = ax[1], norm_hist = True, bins = np.arange(met_mu - 3*met_sig, met_mu + 3*met_sig), kde = False)
ax[1].plot(x1, stats.norm.pdf(x1, met_mu, met_sig),color = '#4394D8')


sb.distplot(np.array(the_rands)[:,1], ax = ax[0], norm_hist = True, bins = np.arange(mincov_mu - 3*mincov_sig, mincov_mu + 3*mincov_sig,(1/len(real_outs))), kde = False)
ax[0].plot(x2, stats.norm.pdf(x2, mincov_mu, mincov_sig),color = '#4394D8')



ax[1].set_title('Number of Metabolites Needed',fontsize = 15)
ax[0].set_title('Coverage of Results',fontsize = 15)
fig.text(0.7,0.89,'Number of random models: '+ str(len(the_rands)),fontsize = 13)
fig.savefig('RandomModels')
plt.close()
# plt.show()
