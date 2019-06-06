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
#Creates a DataFrame of the reported k&r values from the Gore paper (mono_params_gore)
#Creates a DataFrame of the reported parameters from the Gore paper (interacts_gore_unscld)
#Creates a DataFrame of the reported parameters after model rescaling (interacts_gore)
#Creates a DataFrame of the pair model analysis (pair_out_gore)
#Creates a DataFrame of the reported and rescaled parameters adjusted to match pair observations (interacts_gore_pf)
#Creates a DataFrame of the single species timecourse experiments (mono_data)
#Creates a DataFrame of the pair experiment timecourse data (pair_data)
#Creates a DataFrame of the trio experiment (only start/end) (trio_data)
#Creates a DataFrame of the group experiment data (only start/end) (groups_data)

#we have some survival if we choose all our kj1, dj in [1,2].
f1 = 1
dst = 0.49

#first - any big winners? Let's start with those.
[(i,sum([(i in res and not '-' in res) for res in pair_outs_gore.Observed.values])) for i in indx]

k1ea = 1.359768499623295  ###some of these were randomly chosen - the ones that have more than 2 digits
dea =  1.866050260587956

k1pa = 1.6126176171726603
dpa = 1.7079860888998644

k1pch = 2
dpch = 1.1

k1pci = 1.1
dpci = 1.8

k1pf = 1.4675096100204383
dpf = 1.5903157586437142

k1pp = 1.9
dpp = 1

k1pv = 1.5586471333355223
dpv = 1.8023385998245192

k1sm = 1.2
dsm = 1.9



k1vals = {'Ea':k1ea, 'Pa':k1pa, 'Pch':k1pch, 'Pci':k1pci, 'Pf':k1pf, 'Pp':k1pp, 'Pv':k1pv, 'Sm':k1sm}
dvals = {'Ea':dea, 'Pa':dpa, 'Pch':dpch, 'Pci':dpci, 'Pf':dpf, 'Pp':dpp, 'Pv':dpv, 'Sm':dsm}

#now add whatever cross-feeding or cross poisoning we need.
##Start by figuring out current outcomes
winner_nocross = pd.Series(index = pair_outs_gore.index)
for pr in pair_outs_gore.index:
    winner = pr[np.argmax([k1vals[pr[0]]/dvals[pr[0]],k1vals[pr[1]]/dvals[pr[1]]])]
    winner_nocross[pr] = winner


pair_task = pair_outs_gore.copy()
pair_task['NoXtalk'] = winner_nocross
pair_task
needs = pd.Series(index = pair_task.index)
for pr in pair_task.index:
    if pair_task.loc[[pr],'Observed'].values[0] == pair_task.loc[[pr],'NoXtalk'].values[0]:
        needs[pr] = 0 #needs nothing
    elif '-' in pair_task.loc[[pr],'Observed'].values[0]:
        needs[pr] = 1 #needs cross feeding
    else:
        needs[pr] = 2 #needs cross poisoning


pair_task['Needs'] = needs

###Now get k12 and k22 (a and b are 0/1) - at most one of these will be nonzero.
### Model is
#
# dx1/dt = k11x1y1 - d1x1 + k12x1y2
# dx2/dt = k21x2y1 - d2x2 + k22x2y2
# dy1/dt = f1 - dst y1 -  k11x1y1 - k21x2y1
# dy2/dt = a k11 x1y1 + b k21x2y1 - d2st y2 - k12x1y2 - k22x2y2
#
# if 1 is initial survivor then a = 1, b=0, k12 = 0, k22>(1/(f1-dstd1/k11))(d2 - k21 d1/k11)
# if 2 is initial survivor then a = 0, b=1, k22 = 0, k12>(1/(f1-dstd2/k21))(d1 - k11 d2/k21)

pair_models = pd.DataFrame(index = pair_task.index)
acol = pd.Series(index = pair_models.index)
bcol = pd.Series(index = pair_models.index)
k12col = pd.Series(index = pair_models.index)
k22col = pd.Series(index = pair_models.index)
#, columns = ['a','b','k12','k22'])
for pr in pair_models.index:
    if pair_task.loc[[pr],'Needs'].values[0]:
        survivorind = 0
        d1 = dvals[pr[0]]
        d2 = dvals[pr[1]]
        k11 = k1vals[pr[0]]
        k21 = k1vals[pr[1]]
        if pr[1] == pair_task.loc[[pr],'NoXtalk'].values[0]:
            survivorind = 1
        if survivorind:###so 2 is initial survivor
            acol[pr] =0
            bcol[pr]  = 1
            k12col[pr]  =1+(1/(f1-(dst*d2)/k21))*(d1 - k11*d2/k21)
            k22col[pr] = 0
        else:#so 1 is
            acol[pr] =1
            bcol[pr]  = 0
            k12col[pr]  = 0
            k22col[pr] = 1+ (1/(f1-dst*d1/k11))*(d2 - k21*d1/k11)
    else:
            acol[pr] = 0
            bcol[pr]  = 0
            k12col[pr]  =  0
            k22col[pr] = 0



pair_models['a'] = acol
pair_models['b'] = bcol
pair_models['k12'] = k12col
pair_models['k22'] = k22col

# pair model parameters are here and in k1vals,dvals. k1vals give the ki1.
pair_task
pair_models
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

pair_model_outcomes = pd.Series(index = pair_task.index)

for pr in pair_task.index:
    pair_mod = ode(run_pair_mod)
    pair_mod.set_initial_value([1,1,1,0],0).set_f_params([k1vals[pr[0]],k1vals[pr[1]],dvals[pr[0]],dvals[pr[1]]] + list(pair_models.loc[[pr]].values[0]))
    while pair_mod.successful() and pair_mod.t < 100:
        result = pair_mod.integrate(pair_mod.t + 1)
    pair_model_outcomes[pr] = np.array(pr)[result[:2].round(10).astype('bool')]

pair_task['Xtalk'] = pair_model_outcomes
pair_task#it worked





model_net = pd.DataFrame(columns =['Source','Target','Wght','Stype','Etype'])
for i in indx:
    model_net = model_net.append(pd.DataFrame([[i,'Y1',-k1vals[i],'S',-1],['Y1',i,k1vals[i],'R1',1]],columns = ['Source','Target','Wght','Stype','Etype']))


j = 2
for pr in pair_models.index:
    if pair_models.loc[[pr],'a'][0] !=0:
        model_net = model_net.append(pd.DataFrame([[pr[0],'Y' + str(j),k1vals[pr[0]],'S',1]],columns = ['Source','Target','Wght','Stype','Etype']))
    elif pair_models.loc[[pr],'b'][0] !=0:
        model_net = model_net.append(pd.DataFrame([[pr[1],'Y' + str(j),k1vals[pr[1]],'S',1]],columns = ['Source','Target','Wght','Stype','Etype']))
    if pair_models.loc[[pr],'k12'][0] !=0:
        model_net = model_net.append(pd.DataFrame([[pr[0],'Y' + str(j),-abs(pair_models.loc[[pr],'k12'][0]) ,'S',-1],['Y' + str(j),pr[0],pair_models.loc[[pr],'k12'][0],'R2',np.sign(pair_models.loc[[pr],'k12'][0])]],columns = ['Source','Target','Wght','Stype','Etype']))
    if pair_models.loc[[pr],'k22'][0] !=0:
        model_net = model_net.append(pd.DataFrame([[pr[1],'Y' + str(j),-abs(pair_models.loc[[pr],'k22'][0]) ,'S',-1],['Y' + str(j),pr[1],pair_models.loc[[pr],'k22'][0],'R2',np.sign(pair_models.loc[[pr],'k22'][0])]],columns = ['Source','Target','Wght','Stype','Etype']))
    j +=1


model_net


#next need to do the trios, which starts by finding out where we're at for all the trios without addind
## any XTalk. That is
### Model is
#
# dx1/dt = k11*x1*y1 - d1*x1 + k12*x1*y2 + k13*x1*y3
# dx2/dt = k21*x2*y1 - d2*x2 + k22*x2*y2 + k24*x2*y4
# dx3/dt = k31*x3*y1 - d3*x3 + k33*x3*y3 + k34*x3*y4
# dy1/dt = f1 - dst*y1 -  k11*x1*y1 - k21*x2*y1
# dy2/dt = a1*k11*x1*y1 + b1*k21*x2*y1 - dst*y2 - k12*x1*y2 - k22*x2*y2
# dy3/dt = a2*k11*x1*y1 + b2*k31*x3*y1 - dst*y3 - k13*x1*y3 - k33*x3*y3
# dy4/dt = a3*k21*x2*y1 + b3*k31*x3*y1 - dst*y4 - k24*x2*y4 - k34*x3*y4
trio_model_outcomes = real_outs.copy()
trios_unadjusted = pd.DataFrame(index = real_outs.index)
a1col = pd.Series(index = real_outs.index)
a2col = pd.Series(index = real_outs.index)
a3col = pd.Series(index = real_outs.index)
b1col = pd.Series(index = real_outs.index)
b2col = pd.Series(index = real_outs.index)
b3col = pd.Series(index = real_outs.index)

k11col = pd.Series(index = real_outs.index)
k21col = pd.Series(index = real_outs.index)
k31col = pd.Series(index = real_outs.index)

d1col = pd.Series(index = real_outs.index)
d2col = pd.Series(index = real_outs.index)
d3col = pd.Series(index = real_outs.index)

k12col = pd.Series(index = real_outs.index)
k22col = pd.Series(index = real_outs.index)
k33col = pd.Series(index = real_outs.index)
k13col = pd.Series(index = real_outs.index)
k24col = pd.Series(index = real_outs.index)
k34col = pd.Series(index = real_outs.index)

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


def run_trio_mod_unadj(t,y,parms):
    [x1,x2,x3,y1,y2,y3,y4] = y
    [d1,d2,d3,k11,k21,k31,a1,a2,a3,b1,b2,b3,k12,k22,k33,k13,k24,k34] = parms# list(trios_unadjusted.loc[[tr]].values[0])
    dx1dt = k11*x1*y1 - d1*x1 + k12*x1*y2 + k13*x1*y3
    dx2dt = k21*x2*y1 - d2*x2 + k22*x2*y2 + k24*x2*y4
    dx3dt = k31*x3*y1 - d3*x3 + k33*x3*y3 + k34*x3*y4
    dy1dt = f1 - dst*y1 -  k11*x1*y1 - k21*x2*y1
    dy2dt = a1*k11*x1*y1 + b1*k21*x2*y1 - dst*y2 - k12*x1*y2 - k22*x2*y2
    dy3dt = a2*k11*x1*y1 + b2*k31*x3*y1 - dst*y3 - k13*x1*y3 - k33*x3*y3
    dy4dt = a3*k21*x2*y1 + b3*k31*x3*y1 - dst*y4 - k24*x2*y4 - k34*x3*y4
    return [dx1dt,dx2dt,dx3dt,dy1dt,dy2dt,dy3dt,dy4dt]



trio_model_outcomes_unadj = pd.Series(index = real_outs.index)

for tr in real_outs.index:
    tr_mod = ode(run_trio_mod_unadj)
    tr_mod.set_initial_value([1,1,1,1,0,0,0],0).set_f_params(list(trios_unadjusted.loc[[tr]].values[0]))
    while tr_mod.successful() and tr_mod.t < 100:
        result = tr_mod.integrate(tr_mod.t + 1)
    trio_model_outcomes_unadj[tr] = np.array(tr)[result[:3].round(10).astype('bool')]



trio_model_outcomes['NoAddedXtalk'] = trio_model_outcomes_unadj
trio_tasks = pd.Series(index = trio_model_outcomes.index)
for tr in trio_model_outcomes.index:
    obsv = (tr[0] in trio_model_outcomes.loc[[tr],'Observed'].values[0],tr[1] in trio_model_outcomes.loc[[tr],'Observed'].values[0],tr[2] in trio_model_outcomes.loc[[tr],'Observed'].values[0])
    noxt = (tr[0] in trio_model_outcomes.loc[[tr],'NoAddedXtalk'].values[0],tr[1] in trio_model_outcomes.loc[[tr],'NoAddedXtalk'].values[0],tr[2] in trio_model_outcomes.loc[[tr],'NoAddedXtalk'].values[0])
    if obsv == noxt:
        trio_tasks[tr] = 0 #already agrees
    elif sum(obsv) == 3:
        if sum(noxt) == 1:
            trio_tasks[tr] = 13#crossfeed 2 species
        else:
            trio_tasks[tr] = 23#crossfeed 1 species
    elif sum(obsv) == 2:
        if sum(noxt) == 1:
            trio_tasks[tr] = 12#crossfeed 1 species
        elif sum(noxt) == 2:
            trio_tasks[tr] = 22#feed one and poison one...or figure something better out.
        else:
            trio_tasks[tr] = 32#poison one species
    elif sum(obsv) == 1:
        if sum(noxt) == 1:
            trio_tasks[tr] = 11#feed one and poison one...or figure something better out.
        elif sum(noxt) == 2:
            trio_tasks[tr] = 21#poison 1
        else:
            trio_tasks[tr] = 31#poison two species

trio_model_outcomes['Needs'] = trio_tasks

trio_model_outcomes

sum(trio_tasks.values.astype('bool'))#### Number of fixes we need to impliment.


tsk_counts = pd.Series([sum(trio_tasks.values == ii) for ii in [13,23,12,22,32,11,21,31]], index = [13,23,12,22,32,11,21,31])

####For the poisonings, the remaining survivors cross feed. Add a byproduct to that which poisons the
## third. Need to make it potent and dilute slower.
## For the cross-feeding, same just change sign
## For the double cross feeding, need a chain. Need signal->food->second food.
## For switch...woof. Those I did in mathematica.

### Model is
#
# dx1/dt = k11*x1*y1 - d1*x1 + k12*x1*y2 + k13*x1*y3 + psi15*x1*y5
# dx2/dt = k21*x2*y1 - d2*x2 + k22*x2*y2 + k24*x2*y4 + psi25*x2*y5
# dx3/dt = k31*x3*y1 - d3*x3 + k33*x3*y3 + k34*x3*y4 + psi35*x3*y5
# dy1/dt = f1 - dst*y1 -  k11*x1*y1 - k21*x2*y1
# dy2/dt = a1*k11*x1*y1 + b1*k21*x2*y1 - dst*y2 - k12*x1*y2 - k22*x2*y2
# dy3/dt = a2*k11*x1*y1 + b2*k31*x3*y1 - dst*y3 - k13*x1*y3 - k33*x3*y3
# dy4/dt = a3*k21*x2*y1 + b3*k31*x3*y1 - dst*y4 - k24*x2*y4 - k34*x3*y4
# dy5/dt = c3*k12*x1*y2 + c2*k13*x1*y3 + c3*k22*x2*y2 + c1*k24*x2*y4 + c2*k33*x3*y3 + c1*k34*x3*y4 - dst*y5 - psi15*x1*y5 - psi25*x2*y5 - psi35*x3*y5


trio_models = trios_unadjusted.copy()

trio_model_outcomes
c1col = pd.Series(index = real_outs.index)
c2col = pd.Series(index = real_outs.index)
c3col = pd.Series(index = real_outs.index)

psi15col = pd.Series(index = real_outs.index)
psi25col = pd.Series(index = real_outs.index)
psi35col = pd.Series(index = real_outs.index)




for tr in real_outs.index:
    obsv = [tr[0] in trio_model_outcomes.loc[[tr],'Observed'].values[0],tr[1] in trio_model_outcomes.loc[[tr],'Observed'].values[0],tr[2] in trio_model_outcomes.loc[[tr],'Observed'].values[0]]
    noxt = [tr[0] in trio_model_outcomes.loc[[tr],'NoAddedXtalk'].values[0],tr[1] in trio_model_outcomes.loc[[tr],'NoAddedXtalk'].values[0],tr[2] in trio_model_outcomes.loc[[tr],'NoAddedXtalk'].values[0]]
    psis = np.zeros(3)
    cs = np.zeros(3)
    if trio_model_outcomes.loc[[tr],'Needs'].values[0] == 32:#need to kill somebody
        kill = np.array([0,1,2])[np.invert(obsv)]#gives index in the trio tuple of the one to kill
        psis[kill] = -10
        cs[kill] = 1
    elif trio_model_outcomes.loc[[tr],'Needs'].values[0]==23:#need to save somebody
        kill = np.array([0,1,2])[np.invert(noxt)]#gives index in the trio tuple of the one to kill
        psis[kill] = 10
        cs[kill] = 1
    [c1col[tr],c2col[tr],c3col[tr]] = list(cs)
    [psi15col[tr],psi25col[tr],psi35col[tr]] = list(psis)

trio_models['c1'] = c1col
trio_models['c2'] = c2col
trio_models['c3'] = c3col

trio_models['psi15'] = psi15col
trio_models['psi25'] = psi25col
trio_models['psi35'] = psi35col

trio_models.columns

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

trio_model_outcomes_xtalk = pd.Series(index = real_outs.index)

for tr in real_outs.index:
    tr_mod = ode(run_trio_mod)
    tr_mod.set_initial_value([1,1,1,1,0,0,0,0],0).set_f_params(list(trio_models.loc[[tr]].values[0]))
    while tr_mod.successful() and tr_mod.t < 100:
        result = tr_mod.integrate(tr_mod.t + 1)
    trio_model_outcomes_xtalk[tr] = np.array(tr)[result[:3].round(10).astype('bool')]


trio_model_outcomes['WithXtalk'] = trio_model_outcomes_xtalk
trio_model_outcomes


trio_worked = pd.Series(index = trio_model_outcomes.index)
for tr in trio_model_outcomes.index:
    obsv = (tr[0] in trio_model_outcomes.loc[[tr],'Observed'].values[0],tr[1] in trio_model_outcomes.loc[[tr],'Observed'].values[0],tr[2] in trio_model_outcomes.loc[[tr],'Observed'].values[0])
    noxt = (tr[0] in trio_model_outcomes.loc[[tr],'WithXtalk'].values[0],tr[1] in trio_model_outcomes.loc[[tr],'WithXtalk'].values[0],tr[2] in trio_model_outcomes.loc[[tr],'WithXtalk'].values[0])
    if obsv == noxt:
        trio_worked[tr] = 0 #already agrees
    elif sum(obsv) == 3:
        if sum(noxt) == 1:
            trio_worked[tr] = 13#crossfeed 2 species
        else:
            trio_worked[tr] = 23#crossfeed 1 species
    elif sum(obsv) == 2:
        if sum(noxt) == 1:
            trio_worked[tr] = 12#crossfeed 1 species
        elif sum(noxt) == 2:
            trio_worked[tr] = 22#feed one and poison one...or figure something better out.
        else:
            trio_worked[tr] = 32#poison one species
    elif sum(obsv) == 1:
        if sum(noxt) == 1:
            trio_worked[tr] = 11#feed one and poison one...or figure something better out.
        elif sum(noxt) == 2:
            trio_worked[tr] = 21#poison 1
        else:
            trio_worked[tr] = 31#poison two species

trio_model_outcomes['WorkedYN'] = trio_worked

sum(trio_worked.values.astype('bool'))#### Number of fixes we need to impliment.


tsk_counts = pd.Series([sum(trio_worked.values == ii) for ii in [13,23,12,22,32,11,21,31]], index = [13,23,12,22,32,11,21,31])


trio_model_outcomes

##### There are two remaining trios to fix. (Pp,Sm,Pv) must go from lone survivor (Pp) to all three surviving
### This can be done with one signalling molecule from x1 to x2, then crossfeeding of x3 by x2 and x3 by x2.
## The model becomes (copied from Mathematica notebook)
#
# x1dt = k11*x1[t]*y1[t] - d1*x1[t];
# x2dt =  k21*x2[t]*y1[t] - d2*x2[t] + k24*x2[t]*y4[t] + k26*x2[t]*y6[t];
# x3dt = k31*x3[t]*y1[t] - d3*x3[t] + k35*x3[t]*y5[t] ;
# y1dt = f1 - dd1*y1[t] - k11*x1[t]*y1[t] - k21*x2[t]*y1[t] - k31*x3[t]*y1[t];
# y4dt = k31*x3[t]*y1[t] - dd1*y4[t] - k24*x2[t]*y4[t];
# s1dt = k11*x1[t]*y1[t] - dd1*s1[t] - ks22*x2[t]*s1[t]; (*Signal molecule from x1 to x2*)
# y5dt =  ks22*x2[t]*s1[t] - dd1*y5[t] - k35*x3[t]*y5[t]; (*Cross-feeding of x3 by x2*)
# y6dt = k35*x3[t]*y5[t] - dd1*y6[t] - k26*x2[t]*y6[t];
##
##Where the new parameters can be chosen ks22 = 10, k35 = 10, k26 = 10 (dd1 == dst)


#### The last is more difficult. It is (Pf,Pch,Pa) must have only Pf survive.
## We can acheive this with a signal chain, where the signal molecule is never used up or diluted.

# x1dt = k11*x1[t]*y1[t] - d1*x1[t] + k13*x1[t]*y3[t];
# x2dt =  k21*x2[t]*y1[t] - d2*x2[t] - k25*x2[t]*y5[t];
# x3dt = k31*x3[t]*y1[t] - d3*x3[t] - k35*x3[t]*y5[t];
# y1dt = f1 - dd1*y1[t] - k11*x1[t]*y1[t] - k21*x2[t]*y1[t] - k31*x3[t]*y1[t];
# y3dt = k31*x3[t]*y1[t] - dd1*y3[t] - k13*x1[t]*y3[t];
# s1dt = k31*x3[t]*y1[t] ;(*- dd1*s1[t] - ks21*x2[t]*s1[t];*)
# s2dt = ks21*x2[t]*s1[t] ;(*- dd1*s2[t] - ks12*x1[t]*s2[t];*)
# y5dt =  ks12*x1[t]*s2[t] - dd1*y5[t] - k25*x2[t]*y5[t] - k35*x3[t]*y5[t];

### our new parameters can be chosen k21 = 10, k212 = 10, k25 = 10, k35 = 10




for tri in trio_models.index:
    x1toy = trio_models.loc[[tri],'c3'][0]*trio_models.loc[[tri],'k12'][0] +trio_models.loc[[tri],'c2'][0]*trio_models.loc[[tri],'k13'][0]
    x2toy = trio_models.loc[[tri],'c3'][0]*trio_models.loc[[tri],'k22'][0] +trio_models.loc[[tri],'c1'][0]*trio_models.loc[[tri],'k24'][0]
    x3toy = trio_models.loc[[tri],'c2'][0]*trio_models.loc[[tri],'k33'][0] +trio_models.loc[[tri],'c1'][0]*trio_models.loc[[tri],'k34'][0]
    model_net = model_net.append(pd.DataFrame([[tri[0],'Y'+str(j), x1toy, 'S',np.sign(x1toy)],[tri[1],'Y'+str(j), x2toy, 'S',np.sign(x2toy)],[tri[2],'Y'+str(j), x3toy, 'S',np.sign(x3toy)],['Y' + str(j),tri[0],trio_models.loc[[tri],'psi15'][0],'R3',np.sign(trio_models.loc[[tri],'psi15'][0])],['Y' + str(j),tri[1],trio_models.loc[[tri],'psi25'][0],'R3',np.sign(trio_models.loc[[tri],'psi25'][0])],['Y' + str(j),tri[2],trio_models.loc[[tri],'psi35'][0],'R3',np.sign(trio_models.loc[[tri],'psi35'][0])]],columns = ['Source','Target','Wght','Stype','Etype']))
    j += 1




sig_chaing = pd.DataFrame([['Pp','S1',k1vals['Pp'],'S',1],['Sm','S1',k1vals['Pp'],'S',-1],['Sm','Y' + str(j+1),10.0,'S',-1],['Sm','Y' + str(j),10.0,'S',1],['Pv','Y' + str(j),10.0,'S',-1],['Pv','Y' + str(j+1),10.0,'S',1],['Y'+str(j),'Pv',10.0,'R3',1],['Y'+str(j+1),'Sm',10.0,'R3',1],['S1','Sm',1,'Sg',0]],columns = ['Source','Target','Wght','Stype','Etype'])


model_net = model_net.append(sig_chaing)

j+=2

sig2 = pd.DataFrame([['Pf','Y'+str(j),10.0,'S',1],['Pa','Y'+str(j),10.0,'S',-1],['Pa','S2',1.0,'S',1],['Pch','Y'+str(j),10.0,'S',-1],['Pa','S3',1.0,'S',1],['Pa','S2',1.0,'S',1],['Y' + str(j),'Pch',10,'R3',-1],['Y' + str(j),'Pa',10,'R3',-1],['S2','Pch',10,'Sg',0],['S3','Pf',10,'Sg',0]],columns = ['Source','Target','Wght','Stype','Etype'])


model_net = model_net.append(sig2)

model_net[model_net.Wght != 0].to_csv('modelnet.csv')
