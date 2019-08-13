#### Collection of the functions I have written for this project.
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
import time
import warnings
import scipy.special as scm
import multiprocessing as mp
import joblib as jb
# import os
# import sys
# import contextlib

from contextlib import redirect_stdout


# def fileno(file_or_fd):
#     fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
#     if not isinstance(fd, int):
#         raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
#     return fd
#
# @contextlib.contextmanager
# def stdout_redirected(to=os.devnull, stdout=None):
#     """
#     https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
#     """
#     if stdout is None:
#        stdout = sys.stdout
#
#     stdout_fd = fileno(stdout)
#     # copy stdout_fd before it is overwritten
#     #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
#     with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
#         stdout.flush()  # flush library buffers that dup2 knows nothing about
#         try:
#             os.dup2(fileno(to), stdout_fd)  # $ exec >&to
#         except ValueError:  # filename
#             with open(to, 'wb') as to_file:
#                 os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
#         try:
#             yield stdout # allow code to be run with the redirected stdout
#         finally:
#             # restore stdout to its previous value
#             #NOTE: dup2 makes stdout_fd inheritable unconditionally
#             stdout.flush()
#             os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

########## Numerically simulating the models (deterministic)
#compute RHS of 2 species model
def twospec_num(t,y,arg):
    '''Evaluate RHS of pair model ODE
    parameters r1,r2,a12,a21 are scalars passed as list of lists: arg =  [[r1,r2],[a12,a21]]
    state variable y = [x1,x2]
    t is a scalar.
    Dependency: None
    '''
    [r1,r2],[al12,al21] = arg
    [x1,x2] = y
    return [r1*x1*(1 - x1 + al12*x2),r2*x2*(1-x2 + al21*x1)]

#compute RHS of 3 species model
def tthreespec_num(t,y,arg):
    '''Evaluate RHS of trio model ODE
    parameters r1,r2,r3,a12,a13,a21,a23,a31,a32 are scalars passed as list of lists: arg =  [[r1,r2,r3],[a12,a13,a21,a23,a31,a32]]
    state variable y = [x1,x2]
    t is a scalar.
    Dependency: None
    '''
    [r1,r2,r3],[al12,al13,al21,al23,al31,al32] = arg
    [x1,x2,x3] = y
    return [r1*x1*(1 - x1 + al12*x2 + al13*x3),r2*x2*(1-x2 + al21*x1 + al32*x3),r3*x3*(1-x3 + al31*x1 + al32*x2)]

#solve model
def solve_vode(func,f_params,init_condit,endtime,jaco = None, jacargs = None, t0 = 0, dt = 0.05, methd = 'adams'):
    '''Solves an ODE using the vode integrator
    method: ‘adams’ or ‘bdf’ Which solver to use, Adams (non-stiff) or BDF (stiff)
    jaco : callable jac(t, y, *jac_args)
    Jacobian of the right-hand side, jac[i,j] = d f[i] / d y[j]
    can return only endpoint by setting dt = endt-t0
    Dependency: scipy.integrate, numpy
    '''
    sl = ode(func, jac = jaco).set_integrator('vode', method = methd)
    sl.set_initial_value(init_condit, t0).set_f_params(f_params)
    if jacargs != None:
        sl.set_jac_params(jacargs)
    solu = [init_condit]
    rt = [t0]
    # while sl.successful() and sl.t < endtime:
    #     solu = solu + [sl.integrate(sl.t + dt)]
    #     rt = rt + [sl.t]
    with open('DVODE_Errors.txt', 'w') as f:
        with redirect_stdout(f):
            with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be errors.
                warnings.simplefilter('error')
                while sl.successful() and sl.t < endtime:
                    try:
                        solu = solu + [sl.integrate(sl.t + dt)]
                        rt = rt + [sl.t]
                    except UserWarning:
                        if methd == 'adams':
                            methd = 'bdf'
                            sl.set_integrator('vode', method = methd)
                            solu = [init_condit]
                            rt = [t0]
                            sl.t = t0
                        else:
                            solu = solu + [np.array([np.nan]*len(solu[-1]))]
                            sl.t += dt
                            rt = rt + [sl.t]
    return np.array(solu),np.array(rt)


########## Numerically simulating the models (Stochastic)
#exact simulation with Gillespie's algorithm
def gillep(sources,products,ka,x0c,endt, N= 1, wholthing = True):
    '''Create a sample trajectory of a stochastic chemical reaction network using the
    next reaction method. Give sources and products as rows of an array.
    Give x0 as concentration, which will  be turned into counts by multiplication by N (if this
    was a classical chemical reaction, N is volume times Avogadro's number). If you just
    want to to enter counts, then do that and don't give an N
    Dependency: numpy, numpy.random
    '''
    x0 = np.round(N*np.array(x0c).astype(float))
    k = np.array(ka).astype(float)*(np.float64(N)**(-np.sum(sources,axis = 1)+1))
    sources = np.array(sources)
    rxnvectors = products - np.array(sources)
    dim = np.size(x0)
    numrxns = np.size(k)
    if any(np.array(np.shape(sources)) != [numrxns,dim]):
        print("sources doesn't match")
        print(np.shape(sources))
        print ([numrxns,dim])
        return 0
    if wholthing:
        solution = np.array([x0])
        t = np.array([0])
        while t[-1]<= endt:
			#calculate the rates
            lams = k*np.prod(scm.comb(solution[-1],sources)*scm.factorial(sources),1)
            if all(lams == 0):
                print('Not Moving')
                t = np.append(t,endt)
                solution = np.append(solution,np.array(solution[-1]))
                break
            else:
                #lambda 0
                lam0 = sum(lams)
                #need two unif[0,1]s
                u = rand(2)
                #then the delta t_k
                delt = np.log(1/u[0])/lam0
                #update time
                t = np.append(t,t[-1]+delt)
                #which is the next reaction? Figure out and update
                lamsums = [sum(lams[:i]) for i in range(1,len(lams)+1)]/lam0
                lamsums = np.append(lamsums,u[1])
                mu = np.where(np.sort(lamsums)==u[1])
                solution = np.append(solution,np.array(solution[-1] + rxnvectors[mu]),axis = 0)
            #let's control for explosion. (I should look at how to do this analytically...(lams too big or something...))
            if lam0 < 0.0001:
                print('Explosion?')
                t = np.append(t,endt)
                solution = np.append(solution, 10*np.array(solution[-1]))
                break
        return [t, solution/N]
    else:
        t = 0
        solution = np.array([x0])
        while t <= endt:
            #calculate the rates
            lams = k*np.prod(scm.comb(solution[-1],sources)*scm.factorial(sources),1)
            if all(lams == 0):
                # print('Not Moving')
                break
            else:
                #lambda 0
                lam0 = sum(lams)
                #need to unif[0,1]s
                u = rand(2)
                #then the delta t_k
                delt = np.log(1/u[0])/lam0
                t = t+delt
                #which is the next reaction? Figure out and update
                lamsums = [sum(lams[:i]) for i in range(1,len(lams)+1)]/lam0
                lamsums = np.append(lamsums,u[1])
                mu = np.where(np.sort(lamsums)==u[1])
                solution = solution + rxnvectors[mu]
        return solution[0]/float(N)

#approximate simulation with tau-leaping
def tau_postleap_anderson(sources, products, rxnrates, x0c,endt, N= 1):
    '''Simulate a stochastic CRN up to time endt using the tau-leaping with postleap checking
    procedure described in anderson2007. Postleap checking is done in such a way as to not introduce
    bias in the simulation, and prevent negative species counts. Give x0c as concentration, which
    will  be turned into counts by multiplication by N. Enter sources and products as rows of
    an np.array
    Dependency: numpy, numpy.random.rand, numpy.random.binomial, numpy.random.poisson,leap_try'''
    x0 = np.round(N*np.array(x0c).astype(float))
    kvals = np.array(rxnrates).astype(float)*(np.float64(N)**(-np.sum(sources,axis = 1)+1))
    sources = np.array(sources)
    rxnvectors = np.array(products) - sources
    dim = np.size(x0)
    numrxns = np.size(kvals)
    if any(np.array(np.shape(sources)) != [numrxns,dim]):
        print("sources doesn't match")
        print(np.shape(sources))
        print ([numrxns,dim])
        return 0
    #initialize:
    t = 0
    lams = kvals*np.prod(scm.comb(x0,sources)*scm.factorial(sources),1)
    Ts = np.zeros(numrxns)
    Cs = np.zeros(numrxns)
    Ss = [np.zeros((1,2)) for i in range(numrxns)]
    p = 0.75
    pst = 0.9
    q = 0.95
    epsi = 0.1
    #Calculate an initial leap size tau:
    if all(lams == 0):
        return x0c
    mus = np.dot(np.transpose(rxnvectors),lams)
    sigs = np.dot(np.transpose(rxnvectors)**2,lams)
    rxn_orders = np.sum(sources, axis = 1)
    rxn_masks = np.array([[src[i] != 0 for src in sources] for i in range(dim)])
    hori = [max(rxn_orders[rm]) for rm in rxn_masks]
    gis = []
    for spec in range(dim):
        max_rxn_orders_mask = np.array([ro == hori[spec] for ro in rxn_orders[rxn_masks[spec]]])
        max_num_of_spec = max([sr[spec] for sr in sources[rxn_masks[spec]][max_rxn_orders_mask]])
        gi = sum([i/(x0[spec] - i) if x0[spec]>i else 0 for i in range(max_num_of_spec)]) + hori[spec]
        gis = gis + [gi]
    tau = min([max(epsi*x0[i]/gis[i],1)/abs(mus[i]) for i in range(dim)]+[max(epsi*x0[i]/gis[i],1)**2/abs(sigs[i]) for i in range(dim)])
    #Now we approximately simulate forward.
    x = x0
    while t <= endt:
        if all(lams == 0):
            break
        #First, check to see if exact simulation would be faster. This is true if our leap is smaller than the time to the actual next jump, so statistically true if tau is not bigger than epsi*1/sum(lams)
        lam0 = sum(lams)
        if tau >= epsi*(1/lam0):
            #use tau-leaping to get the number of times each reaction fires and a row to update the S matrix
            leap1 = [leap_try(Ss[i],lams[i],tau) for i in range(numrxns)]
            #check the leap condition
            fires = np.array([l[1] for l in leap1])
            temp_new_x = x + np.dot(np.transpose(rxnvectors),fires)
            temp_new_lams = kvals*np.prod(scm.comb(temp_new_x,sources)*scm.factorial(sources),1)
            lam_diffs = abs(lams - temp_new_lams)
            satisfy_leap_condition = all([lam_diffs[i]<= max(epsi*lams[i],kvals[i]) for i in range(numrxns)])
            if satisfy_leap_condition:#the leap worked hurray!
                for i in range(numrxns):
                    rwk = int(leap1[i][2])
                    Ss[i] = leap1[i][0][rwk:]
                    Cs[i] = Ss[i][0,1]
                    Ts[i] = Ss[i][0,0]
                t = t + tau
                x = temp_new_x
                lams = temp_new_lams
                satisfy_leap_condition_harsh = all([lam_diffs[i]<= max(0.75*epsi*lams[i],kvals[i]) for i in range(numrxns)])
                if satisfy_leap_condition_harsh:
                    tau = tau**q
                else:
                    tau = tau*pst
            else:#the leap was too too big. Bummer.
                for i in range(numrxns):
                    Ss[i] = leap1[i][0]
                    tau = tau*p
        else:
            #do not use tau leaping. Need to not throw info out though! Using Algorithm 3 of anderson2007:
            #need x,t,lams,Ss,Ts,Cs
            js = np.empty(numrxns)
            for j in range(numrxns):
                js[j] = np.where(Ss[j][:,1]==Cs[j])[0][-1]
            jnotexist = [Cs[j] == Ss[j][-1,1] for j in range(numrxns)]#js[j]!= len(Ss[j])-1
            easy_ps = np.arange(numrxns)[np.array(jnotexist)]
            hard_ps = np.arange(numrxns)[~np.array(jnotexist)]
            ps = np.empty(numrxns)
            for k in easy_ps:
                ps[k] = Ss[k][-1,0]+np.log(1/rand())
            for k in hard_ps:
                tk = Ss[k][int(js[k])+1,0]-Ss[k][int(js[k]),0]
                Nk = Ss[k][int(js[k])+1,1]-Ss[k][int(js[k]),1]
                ps[k] = Ss[k][int(js[k]),0] + tk*(1-rand()**(1/Nk))
            while t<endt and tau < epsi*(1/lam0):
                delts = np.array([(ps[j] - Ts[j])/lams[j] if lams[j]!=0 else np.inf for j in range(len(lams))])
                delta = min(delts)
                mu = np.where(delts == delta)
                t = t+delta
                x = x+rxnvectors[mu][0]
                Ts = Ts + delta*lams
                Cs[mu] += 1
                muval = mu[0][0]
                if Cs[mu]<Ss[muval][-1,1]:
                    jj = np.where(Ss[muval][:,1] > Cs[mu])[0][0]#
                    #max(np.arange(len(Ss[muval]))[np.array([Cs[muval] >= s[1] for s in Ss[muval]])])
                    tk = Ss[muval][jj,0]-max(ps[mu],Ss[muval][jj-1,0])
                    Nk = Ss[muval][jj,1]-max(Cs[mu],Ss[muval][jj-1,1])
                    ps[mu] = max(ps[mu],Ss[muval][jj-1,0]) + tk*(1-rand()**(1/Nk))
                else:
                    ps[mu] = max(Ss[muval][-1,0],ps[mu]) + np.log(1/rand())
                tn_lams = kvals*np.prod(scm.comb(x,sources)*scm.factorial(sources),1)
                lam_diffs = abs(lams - temp_new_lams)
                lam_change = all([lam_diffs[i]<= max(0.5*epsi*lams[i],kvals[i]) for i in range(numrxns)])
                if lam_change:
                    tau = 2*delta
                lams = tn_lams
                lam0 = sum(lams)
                for i in range(numrxns):
                    rok = max(np.arange(len(Ss[i]))[np.array([Ts[i] >= s[0] for s in Ss[i]])])
                    Ss[i] = np.insert(Ss[i],rok+1,[Ts[i],Cs[i]],axis = 0)
                    Ss[i] = Ss[i][rok+1:]
    return x/float(N)


#internal (leap) to tau_postleap_anderson
def leap_try(S,lam,tau):
    '''Leaping for tau_postleap_anderson
    Dependency: numpy, numpy.random.rand, numpy.random.binomial, numpy.random.poisson,'''
    Tk = S[0,0]
    Ck = S[0,1]
    Bk = len(S)-1
    ltime = lam*tau + Tk
    if ltime >= S[-1,0]:
        Nk = poisson(Tk+lam*tau-S[-1,0]) + S[-1,1] - Ck
        rowk = Bk
    else:
        internal_times = S[:,0]
        Ik = np.arange(len(S)-1)[np.array([internal_times[i]<= ltime and internal_times[i+1] > ltime for i in range(len(internal_times)-1)])][0]
        r = (Tk + lam*tau - S[Ik,0])/(S[Ik+1,0]-S[Ik,0])
        Nk = binomial(int(S[Ik+1,1]-S[Ik,1]),r) + S[Ik,1] - Ck
        rowk = Ik
    newS = np.insert(S,rowk+1,[ltime,Ck+Nk],axis = 0)
    # print(newS)
    return [newS,Nk,rowk+1]

#convert interaction & r-value tables into parameters for Gillespie's algorithm
def make_k_pair(inters,rvals, pair):
    '''Using DF of all interaction parameters, construct a vector of reaction rates
    X <->2X, Y<->2Y, X+Y -> 2X+Y (beta_12>0), X+Y->Y (beta_12<0), X+Y -> X+2Y (beta_21>0), X+Y -> X (beta_21 < 0)
    in that order. That is r1,-r1,r2,-r2,(b12 <0),(b12<0),(b21 > 0),(b21 < 0)
    To be perfectly sure of the order, let's make a dict.
    pair can be passed as any subscribtable object (tuple, list, array)
    rvals should be pd.Series
    Dependency: pandas, numpy
    '''
    k={}
    k['X->2x'] = rvals.loc[pair[0]]
    k['2X->X'] = rvals.loc[pair[0]]
    k['Y->2Y'] = rvals.loc[pair[1]]
    k['2Y->Y'] = rvals.loc[pair[1]]
    b12 = inters.loc[pair[0],pair[1]]
    if b12 > 0:
        k['X+Y->2X+Y'] = b12*rvals.loc[pair[0]]
        k['X+Y->Y'] = 0
    else:
        k['X+Y->2X+Y'] = 0
        k['X+Y->Y'] = np.abs(b12*rvals.loc[pair[0]])
    b21 = inters.loc[pair[1],pair[0]]
    if b21 > 0:
        k['X+Y->X+2Y'] = b21*rvals.loc[pair[1]]
        k['X+Y->X'] = 0
    else:
        k['X+Y->X+2Y'] = 0
        k['X+Y->X'] = np.abs(b21*rvals.loc[pair[1]])
    kvals = np.array([k['X->2x'],k['2X->X'],k['Y->2Y'],k['2Y->Y'],k['X+Y->2X+Y'],k['X+Y->Y'],k['X+Y->X+2Y'],k['X+Y->X']])
    return kvals

#convert interaction & r-value tables into parameters for Gillespie's algorithm
def make_k_trio(inters,rvals,trio):
        '''Using DF of all interaction parameters, construct a vector of reaction rates
        # X->2X, 2X->X, Y->2Y, 2Y->Y, Z->2Z, 2Z->Z
        # X+Y->2X+Y (beta_12>0), X+Y->Y (beta_12<0), X+Z->2X+Z (beta_13>0), X+Z->Z (beta_13<0)
        # X+Y->X+2Y (beta_21>0), X+Y->X (beta_21<0), Y+Z->2Y+Z (beta_23>0), Y+Z->Z (beta_23<0)
        # X+Z->X+2Z (beta_31>0), X+Z->X (beta_31<0), Y+Z->Y+2Z (beta_32>0), Y+Z->Y (beta_32<0)
        in that order. That is r1,-r1,r2,-r2,(b12 <0),(b12<0),(b21 > 0),(b21 < 0)
        To be perfectly sure of the order, let's make a dict.
        trio can be passed as any subscribtable object (tuple, list, array)
        rvals should be pd.Series
        Dependency: pandas, numpy
        '''
        k={}
        k['X->2x'] = rvals.loc[trio[0]]
        k['2X->X'] = rvals.loc[trio[0]]
        k['Y->2Y'] = rvals.loc[trio[1]]
        k['2Y->Y'] = rvals.loc[trio[1]]
        k['Z->2Z'] = rvals.loc[trio[2]]
        k['2Z->Z'] = rvals.loc[trio[2]]
        b12 = inters.loc[trio[0],trio[1]]
        if b12 > 0:
            k['X+Y->2X+Y'] = b12*rvals.loc[trio[0]]
            k['X+Y->Y'] = 0
        else:
            k['X+Y->2X+Y'] = 0
            k['X+Y->Y'] = np.abs(b12*rvals.loc[trio[0]])
        b13 = inters.loc[trio[0],trio[2]]
        if b13 > 0:
            k['X+Z->2X+Z'] = b13*rvals.loc[trio[0]]
            k['X+Z->Z'] = 0
        else:
            k['X+Z->2X+Z'] = 0
            k['X+Z->Z'] = np.abs(b13*rvals.loc[trio[0]])
        b21 = inters.loc[trio[1],trio[0]]
        if b21 > 0:
            k['X+Y->X+2Y'] = b21*rvals.loc[trio[1]]
            k['X+Y->X'] = 0
        else:
            k['X+Y->X+2Y'] = 0
            k['X+Y->X'] = np.abs(b21*rvals.loc[trio[1]])
        b23 = inters.loc[trio[1],trio[2]]
        if b23 > 0:
            k['Y+Z->2Y+Z'] = b23*rvals.loc[trio[1]]
            k['Y+Z->Z'] = 0
        else:
            k['Y+Z->2Y+Z'] = 0
            k['Y+Z->Z'] = np.abs(b23*rvals.loc[trio[1]])
        b31 = inters.loc[trio[2],trio[0]]
        if b31 > 0:
            k['X+Z->X+2Z'] = b31*rvals.loc[trio[2]]
            k['X+Z->X'] = 0
        else:
            k['X+Z->X+2Z'] = 0
            k['X+Z->X'] = np.abs(b31*rvals.loc[trio[2]])
        b32 = inters.loc[trio[2],trio[1]]
        if b32 > 0:
            k['Y+Z->Y+2Z'] = b32*rvals.loc[trio[2]]
            k['Y+Z->Y'] = 0
        else:
            k['Y+Z->Y+2Z'] = 0
            k['Y+Z->Y'] = np.abs(b32*rvals.loc[trio[2]])
        return np.array([k['X->2x'],k['2X->X'],k['Y->2Y'],k['2Y->Y'],k['Z->2Z'],k['2Z->Z'],k['X+Y->2X+Y'],k['X+Y->Y'],k['X+Z->2X+Z'], k['X+Z->Z'],k['X+Y->X+2Y'],k['X+Y->X'],k['Y+Z->2Y+Z'],k['Y+Z->Z'],k['X+Z->X+2Z'],k['X+Z->X'],k['Y+Z->Y+2Z'],k['Y+Z->Y']])


########## Finding and assessing the stability of fixed points:
##Compute equilibria of 3 species model
def threesp_eq(R,B):
    '''Return list of dicts - the 8 equilibiria of the three species model.
    R: intrinsic growth parameters
    B: interaction parameters
    Dependency: numpy
    '''
    r1,r2,r3 = R
    b12,b13,b21,b23,b31,b32 = B
    ss1 = {'x1':0.0,'x2':0.0,'x3':0.0}
    ss2 = {'x1':1.0,'x2':0.0,'x3':0.0}
    ss3 = {'x1':0.0,'x2':1.0,'x3':0.0}
    ss4 = {'x1':0.0,'x2':0.0,'x3':1.0}
    if b12*b21 != 1.0:
        ss5 = {'x1':(1.0+b12)/(1-b12*b21), 'x2':(1.0+b21)/(1-b12*b21), 'x3':0}
    else:
        ss5 = 'fail'
    if b13*b31 != 1:
        ss6 = {'x1':(1.0+b13)/(1-b13*b31), 'x2':0, 'x3':(1.0+b31)/(1-b13*b31)}
    else:
        ss6 = 'fail'
    if b23*b32 != 1:
        ss7 = {'x1':0, 'x2':(1.0+b23)/(1-b23*b32), 'x3':(1.0+b32)/(1-b23*b32)}
    else:
        ss7 = 'fail'
    bmat = np.array([[1,-b12,-b13],[-b21,1,-b23],[-b31,-b32,-1]])
    try:
        last_ss = np.linalg.solve(bmat,[1,1,1])
        ss8 = {'x1':last_ss[0],'x2':last_ss[1],'x3':last_ss[2]}
    except LinAlgErr as err:
        ss8 = 'fail'
    return [ss1,ss2,ss3,ss4,ss5,ss6,ss7,ss8]

##Compute jacobian matrix of 3 species model
def jac_thr(X,R,B):
    '''The jacobian of the three species model
    X: state variable
    R: intrinsic growth parameters
    B: interaction parameters
    Dependency: numpy'''
    x1,x2,x3 = X
    r1,r2,r3 = R
    b12,b13,b21,b23,b31,b32 = B
    jac_row1 = np.array([r1*(1-2*x1+b12*x2+b13*x3), r1*b12*x1, r1*b13*x1])
    jac_row2 = np.array([r2*b21*x2, r2*(1-2*x2+b21*x1+b23*x3), r2*b23*x2])
    jac_row3 = np.array([r3*b31*x3, r3*b32*x3, r3*(1-2*x3+b31*x1+b32*x2)])
    return np.array([jac_row1,jac_row2,jac_row3])

##Compare stability of the model with outcome for a single trio
def test_trio_LV(tr, trio_interacts,kvals, rvals,trio_observation,pair_observation, details = True, allow_bistab = True, searching = False):
    '''Determine if a parameter set explains a trio
    tr - trio as tuple
    trio_interacts DF of interactions (can be just from trio or all it shouldn't matter)
    kvals - carrying capacities as dictionary
    rvals - growth rates as dictionary
    trio_observation - outcome of trio as string 'Xx' or 'Xx-Xx' or 'Xx-Xx-Xx'
    pair_observation - outcome of the 3 pair experiments involved in the trio as DataFrame
    Dependency: pandas, numpy, jac_thr,threesp_eq
    '''
    pair_outcomes = pair_observation.copy()
    ### For each pair, we actually don't need to solve anything or find eigenvalues, etc, we can just look at the (rescaled)
    ### parameter values. For triplets, we can assess stability of the double and single extinction steady states
    ### without much work, but the positive steady state will require some calculation

    ### figure out what the parameters predict for each pair
    lv_pre  = pd.Series(index = pair_outcomes.index)
    for ii in lv_pre.index:
        tf = np.array([trio_interacts.loc[ii[0],ii[1]],trio_interacts.loc[ii[1],ii[0]]]) > -1
        lv_pre.loc[ii] = '-'.join(np.array(ii)[np.where(tf)])

    pair_outcomes.loc[:,'LVPrediction'] = lv_pre

    ## and if it's correct
    pair_outcomes.loc[:,'LVRight'] = pair_outcomes.loc[:,'Observed'] == pair_outcomes.loc[:,'LVPrediction']

    #outcome will contain all the details of the analysis of the 3 species model.
    outcome = pd.DataFrame(index = ['0',tr[0],tr[1],tr[2],'-'.join([tr[0],tr[1]]),'-'.join([tr[0],tr[2]]),'-'.join([tr[1],tr[2]]),'-'.join([tr[0],tr[1],tr[2]])],columns = ['value','eigenvalues','stability','center'])
    steady_states = threesp_eq([rvals[tr[0]],rvals[tr[1]],rvals[tr[0]]],[trio_interacts.loc[tr[0],tr[1]],trio_interacts.loc[tr[0],tr[2]],trio_interacts.loc[tr[1],tr[0]],trio_interacts.loc[tr[1],tr[2]],trio_interacts.loc[tr[2],tr[0]],trio_interacts.loc[tr[2],tr[1]]])
    for eq in range(8):
        if steady_states[eq] != 'fail':
            px1 = steady_states[eq]['x1']
            px2 = steady_states[eq]['x2']
            px3 = steady_states[eq]['x3']
            outcome.iloc[eq,0] = (px1,px2,px3)
            jactemp = jac_thr([px1,px2,px3],[rvals[tr[0]],rvals[tr[1]],rvals[tr[0]]],[trio_interacts.loc[tr[0],tr[1]],trio_interacts.loc[tr[0],tr[2]],trio_interacts.loc[tr[1],tr[0]],trio_interacts.loc[tr[1],tr[2]],trio_interacts.loc[tr[2],tr[0]],trio_interacts.loc[tr[2],tr[1]]])
            eigst = np.linalg.eig(jactemp.astype(float))[0].real
            eigs = np.array([eg if abs(eg) > 10**(-5) else 0 for eg in eigst])
            outcome.iloc[eq,1] = eigs
            outcome.iloc[eq,2] = sum(eigs <= 0)
            outcome.iloc[eq,3] = bool(sum(eigs == 0))
        else:
            outcome.iloc[eq,1] = list(outcome.iloc[eq,0])
            outcome.iloc[eq,2] = 0
            outcome.iloc[eq,3] = False
    tio_predictions_summ = ' & '.join(list(outcome.index[np.where(outcome.stability == 3)]))
    if tio_predictions_summ == '':
        tio_predictions_summ = 'Blow Up'

    if allow_bistab:
        bistab = False #just so it doesnt have an effect on got_it.
        trio_correct = set([trio_observation]).issubset(tio_predictions_summ.split(' & '))

    else:
        bistab = '&' in tio_predictions_summ
        trio_correct = trio_observation == tio_predictions_summ

    pair_correct = sum(pair_outcomes.LVRight)/len(pair_outcomes)==1

    prod = (trio_interacts.values * np.transpose(trio_interacts.values) < 1)
    no_unbounded = all(trio_interacts.values[np.where(prod >1)] < -1)

    got_it = all([no_unbounded,pair_correct,trio_correct,not bistab])
    if searching:
        return got_it,pair_correct,outcome.loc[:,'eigenvalues'], outcome.loc[:,'stability'] == 3
    else:
        if details:
            return got_it,pair_outcomes,tio_predictions_summ,outcome
        else:
            return got_it

##Compare stability of the model with outcome for all trios
def test_params_LV(interacts, kvals, rvals, trio_observation, pair_observation, details = True, allow_bistab = True, searching = False):
    '''Tests a set of parameters for whether or not they match with the outcomes
    of the experiments, according to the Lotka-Volterra model.
    interacts - DF of interaction parameters with index/labels the species names.
    kvals - carrying capacities as dictionary
    rvals - growth rates as dictionary
    trio_observation - DF of outcomes of the trio experiments
    pair_observation - DF of outcomes of the pair experiments
    Dependency: numpy, pandas,jac_thr,threesp_eq
    '''

    real_outcomes = trio_observation.copy()
    pair_outcomes = pair_observation.copy()


    ### For each pair, we actually don't need to solve anything or find eigenvalues, etc, we can just look at the (rescaled)
    ### parameter values. For triplets, we can assess stability of the double and single extinction steady states
    ### without much work, but the positive steady state will require some calculation

    interacts2 = interacts.copy()


    prod = interacts2.values * np.transpose(interacts2.values)
    no_unbounded = sum(sum((prod <1).astype('int') + np.eye(len(prod)))) - len(prod)**2 == 0
    #above shows that every pair has product of interaction terms less than 1. Therefore
    # we only need to test if one or both is less than -1.
    #(interacts2 >-1).astype('int') + (interacts2 >-1).astype('int').transpose()
    #No bistability - one means we have an extinction, 2 means we have positive steady state.
    #If there is extinction, suppose b_{ij}<-1. Then i loses to j.
    #(interacts <-1).astype('int')



    ##### Triplet outcomes - produces a dictionary with the winners (_summ), and with the full outcomes (_full), which has
    ##### all equilibrium and eigenvalues of the Jacobian. These are as predicted with the gLV model using the
    ##### paramters reported in the paper.
    tio_predictions_full = {}
    tio_predictions_summ = {}
    for tr in real_outcomes.index:
        outcome = pd.DataFrame(index = ['0',tr[0],tr[1],tr[2],'-'.join([tr[0],tr[1]]),'-'.join([tr[0],tr[2]]),'-'.join([tr[1],tr[2]]),'-'.join([tr[0],tr[1],tr[2]])],columns = ['value','eigenvalues','stability','center'])
        steady_states = threesp_eq([rvals[tr[0]],rvals[tr[1]],rvals[tr[0]]],[interacts2.loc[tr[0],tr[1]],interacts2.loc[tr[0],tr[2]],interacts2.loc[tr[1],tr[0]],interacts2.loc[tr[1],tr[2]],interacts2.loc[tr[2],tr[0]],interacts2.loc[tr[2],tr[1]]])
        for eq in range(8):
            if steady_states[eq] != 'fail':
                px1 = steady_states[eq]['x1']
                px2 = steady_states[eq]['x2']
                px3 = steady_states[eq]['x3']
                outcome.iloc[eq,0] = (px1,px2,px3)
                jactemp = jac_thr([px1,px2,px3],[rvals[tr[0]],rvals[tr[1]],rvals[tr[0]]],[interacts2.loc[tr[0],tr[1]],interacts2.loc[tr[0],tr[2]],interacts2.loc[tr[1],tr[0]],interacts2.loc[tr[1],tr[2]],interacts2.loc[tr[2],tr[0]],interacts2.loc[tr[2],tr[1]]])
                eigs = np.linalg.eig(np.array(jactemp).astype(float))[0].real
                outcome.iloc[eq,1] = eigs
                outcome.iloc[eq,2] = sum(eigs <= 0)
                outcome.iloc[eq,3] = bool(sum(eigs == 0))
            else:
                outcome.iloc[eq,1] = list(outcome.iloc[eq,0])
                outcome.iloc[eq,2] = 0
                outcome.iloc[eq,3] = False
        tio_predictions_full[tr] = outcome
        tio_predictions_summ[tr] = ' & '.join(list(outcome.index[np.where(outcome.stability == 3)]))
        if tio_predictions_summ[tr] == '':
            tio_predictions_summ[tr] = 'Blow Up'

    #add the gLV predicted outcomes to the DF with real outcomes in it.
    glv = pd.Series(index = real_outcomes.index)
    for i in glv.index:
        glv.loc[i] = tio_predictions_summ[i]


    all_outcomes = real_outcomes
    all_outcomes.loc[:,'gLV Predictions'] = glv

    all_outcomes.loc[:,'gLVRight'] = all_outcomes.loc[:,'Observed'] == all_outcomes.loc[:,'gLV Predictions']

    all_outcomes.loc[:,'Bistable'] = ['&' in ou for ou in all_outcomes.loc[:,'gLV Predictions']]
    if allow_bistab:
        for tri in all_outcomes[all_outcomes.loc[:,'Bistable']].index:
            all_outcomes.loc[[tri],'gLVRight'] = {all_outcomes.loc[[tri],'Observed'][0]}.issubset(all_outcomes.loc[[tri],'gLV Predictions'][0].split(' & '))

    #all_outcomes
    trio_correct = sum(all_outcomes.gLVRight)/len(all_outcomes) ==1

    bistability = sum(all_outcomes.Bistable)
    # DataFrame for pair competitions, real and as predicted with the gLV model. There
    # is no table in the paper, so these are inferred from the plotted trajectories in the
    # supplimental. They say there are 10 instances of extinction.
    lv_pre  = pd.Series(index = pair_outcomes.index)

    for ii in lv_pre.index:
        tf = np.array([interacts2.loc[ii[0],ii[1]],interacts2.loc[ii[1],ii[0]]]) > -1
        lv_pre.loc[ii] = '-'.join(np.array(ii)[np.where(tf)])

    pair_outcomes.loc[:,'LVPrediction'] = lv_pre

    pair_outcomes.loc[:,'LVRight'] = pair_outcomes.loc[:,'Observed'] == pair_outcomes.loc[:,'LVPrediction']

    pair_correct = sum(pair_outcomes.LVRight)/len(pair_outcomes)==1

    got_it = all([no_unbounded,pair_correct,trio_correct,not bistability])

    if searching:
        return got_it,pair_correct,all_outcomes,tio_predictions_full
    else:
        if details:
            return got_it,pair_outcomes,all_outcomes,tio_predictions_full
        else:
            return got_it

##Compute single extinction state for 3 species model
def get_singext_state(sp1,sp2,interactions):
    '''Compute the value of the fixed point on the sp1/sp2 plane
    Dependency: pandas
    '''
    b12 = interactions.loc[sp1,sp2]
    b21 = interactions.loc[sp2,sp1]
    return ((1+b12)/(1-b12*b21),(1+b21)/(1-b12*b21))


########### Finding parameters with desired stability result
##Compute 3rd eigenvalue for single extinction state
def lam3(blist):
    '''Compute third eigenvalue of the fixed point on the sp1/sp2 plane, given list of
    paramters
    Dependecy: None'''
    b12,b21,b31,b32 = blist
    return b31*(1+b12)/(1-b12*b21) + b32*(1+b21)/(1-b12*b21) + 1

##Compute 3rd eigenvalue for single extinction state
def get_lambda3(sp,interactions):
    '''Compute third eigenvalue of the fixed point on the sp1/sp2 plane,
    given the species names and DF of all species interaction parameters.
    Dependecy: pandas'''
    sp1,sp2,sp3 = sp
    b12 = interactions.loc[sp1,sp2]
    b21 = interactions.loc[sp2,sp1]
    b31 = interactions.loc[sp3,sp1]
    b32 = interactions.loc[sp3,sp2]
    return b31*(1+b12)/(1-b12*b21) + b32*(1+b21)/(1-b12*b21) + 1

##Distance to variety l3 = 0
def g(x_list,b_list):
    '''Compute distance to l3=0 where l3 is the third eigenvalue of the fixed point on the sp1/sp2 plane
    Dependecy: None'''
    bt12,bt21,bt31,bt32 = b_list
    s,t,q = x_list
    return (bt12 - (s-1))**2 + (bt21 - (t-1))**2 + (bt31 - (q*t - 1 + t/2))**2 + (bt32 - ( -s*q - 1 +s/2))**2

##Parameterization of variety l3 = 0
def vari(x_list):
    '''Return a point on the variety l3 = 0 in parameter space, where  l3 is the third eigenvalue of the fixed point on the sp1/sp2 plane
    Dependency: numpy'''
    s,t,q = x_list
    return np.array([s-1,t-1,q*t - 1 + t/2,-s*q - 1 +s/2])

##Change sign of l3
def change_stability(trio,state,interactions,gfun, un = True,differ = 0.001):
    '''Provides options to "unstabilize" or "stabilize" the single extinction state
    by only changing one parameter or all of them (minimizing the change). This assumes
    that only the third eigenvalue needs to be changed.
    Dependecy: numpy, pandas, numpy.random, scipy.optimize, vari'''
    tflist = [a in state.split('-') for a in trio]
    ftlist = [not a for a in tflist]
    spe1 = trio[np.arange(3)[tflist][0]]
    spe2 = trio[np.arange(3)[tflist][1]]
    spe3 = trio[np.arange(3)[ftlist][0]]
    b12 = interactions.loc[spe1,spe2]
    b21 = interactions.loc[spe2,spe1]
    b31 = interactions.loc[spe3,spe1]
    b32 = interactions.loc[spe3,spe2]
    options = {'old':[b12,b21,b31,b32]}
    b31_new = (b21*b12-1-b32*(1+b21))/(1+b12) #makes lambda3 = 0
    if (1+b12)>0:
        if not un:
            options['.'.join(['b',spe3,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31_new - differ,(spe3,spe2):b32}
        else:
            options['.'.join(['b',spe3,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31_new + differ,(spe3,spe2):b32}
    else:
        if not un:
            options['.'.join(['b',spe3,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31_new + differ,(spe3,spe2):b32}
        else:
            options['.'.join(['b',spe3,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31_new - differ,(spe3,spe2):b32}
    b32_new = (b21*b12-1-b31*(1+b12))/(1+b21) #makes lambda3 = 0
    if (1+b21)>0:
        if not un:
            options['.'.join(['b',spe3,spe2])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32_new - differ}
        else:
            options['.'.join(['b',spe3,spe2])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32_new + differ}
    else:
        if not un:
            options['.'.join(['b',spe3,spe2])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32_new + differ}
        else:
            options['.'.join(['b',spe3,spe2])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32_new - differ}
    b12_new = (1+b31+b32+b21*b32)/(b21-b31) #makes lambda3 = 0
    if (b31-b21)>0:
        if not un:
            options['.'.join(['b',spe1,spe2])] = {(spe1,spe2):b12_new - differ,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32}
        else:
            options['.'.join(['b',spe1,spe2])] = {(spe1,spe2):b12_new + differ,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32}
    else:
        if not un:
            options['.'.join(['b',spe1,spe2])] = {(spe1,spe2):b12_new + differ,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32}
        else:
            options['.'.join(['b',spe1,spe2])] = {(spe1,spe2):b12_new - differ,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32}
    b21_new = (1+b31+b32+b12*b31)/(b12-b32) #makes lambda3 = 0
    if (b32-b12)>0:
        if not un:
            options['.'.join(['b',spe2,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21_new - differ,(spe3,spe1):b31,(spe3,spe2):b32}
        else:
            options['.'.join(['b',spe2,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21_new + differ,(spe3,spe1):b31,(spe3,spe2):b32}
    else:
        if not un:
            options['.'.join(['b',spe2,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21_new + differ,(spe3,spe1):b31,(spe3,spe2):b32}
        else:
            options['.'.join(['b',spe2,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21_new - differ,(spe3,spe1):b31,(spe3,spe2):b32}
    min_l2 = minimize(gfun, rand(3),[b12,b21,b31,b32])['x']
    pt_on_var = vari(min_l2) #makes lambda3 = 0
    vec =  pt_on_var - np.array([b12,b21,b31,b32]) #the vector from th original paramters to the point on the variety
    vec2 = np.array([b12,b21,b31,b32]) + (1 + differ)*vec #this changes the sign - don't need to choose stable/unstable
    options['l2min'] = {(spe1,spe2):vec2[0],(spe2,spe1):vec2[1],(spe3,spe1):vec2[2],(spe3,spe2):vec[3]}
    return options

##Make l3 = 0
def make_center(trio,state,interactions,gfun,pair_key = ()):
    '''Provides options to make the third eigengalue 0 for an extinction state without changing the interactions between the
    two species in the pair (and also not changing the other eigenvalues sign)
    trio - tuple of trio
    state - string, state to give a center subspace to
    interactions - DataFrame
    gfun - function: distance from 0 variety to betas, choose the right one!
    pair_key - (1,2) means your pair is (X,X, ), (1,3) means (X, ,X), (2,3) means ( ,X,X)
    Dependency: pandas, numpy, numpy.random,scipy.optimize, vari
    '''
    tflist = [a in state.split('-') for a in trio]
    ftlist = [not a for a in tflist]
    spe1 = trio[np.arange(3)[tflist][0]]
    spe2 = trio[np.arange(3)[tflist][1]]
    spe3 = trio[np.arange(3)[ftlist][0]]
    b12 = interactions.loc[spe1,spe2]
    b21 = interactions.loc[spe2,spe1]
    b31 = interactions.loc[spe3,spe1]
    b32 = interactions.loc[spe3,spe2]
    options = {}
    #
    b31_new = (b21*b12-1-b32*(1+b21))/(1+b12) #makes lambda3 = 0
    options['.'.join(['b',spe3,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31_new,(spe3,spe2):b32}
    #
    b32_new = (b21*b12-1-b31*(1+b12))/(1+b21) #makes lambda3 = 0
    options['.'.join(['b',spe3,spe2])] = {(spe1,spe2):b12,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32_new}
    #
    b12_new = (1+b31+b32+b21*b32)/(b21-b31) #makes lambda3 = 0
    options['.'.join(['b',spe1,spe2])] = {(spe1,spe2):b12_new,(spe2,spe1):b21,(spe3,spe1):b31,(spe3,spe2):b32}
    #
    b21_new = (1+b31+b32+b12*b31)/(b12-b32) #makes lambda3 = 0
    options['.'.join(['b',spe2,spe1])] = {(spe1,spe2):b12,(spe2,spe1):b21_new,(spe3,spe1):b31,(spe3,spe2):b32}
    #
    if pair_key == (1,2):
        min_l2 = minimize(gfun, rand(),[b12,b21,b31,b32])['x']
        min_ls = np.append([b12 + 1,b21 + 1],min_l2)
        pt_on_var = vari(min_ls) #makes lambda3 = 0
        options['l2min'] = {(spe1,spe2):pt_on_var[0],(spe2,spe1):pt_on_var[1],(spe3,spe1):pt_on_var[2],(spe3,spe2):pt_on_var[3]}
    elif pair_key == (1,3):
        min_l2 = minimize(gfun, np.random.rand(2),[b12,b21,b31,b32])['x']
        min_ls = np.append(min_l2,1/min_l2[1] - 1/2 + b31/min_l2[1])
        pt_on_var = vari(min_ls) #makes lambda3 = 0
        options['l2min'] = {(spe1,spe2):pt_on_var[0],(spe2,spe1):pt_on_var[1],(spe3,spe1):pt_on_var[2],(spe3,spe2):pt_on_var[3]}
    elif pair_key == (2,3):
        min_l2 = minimize(gfun, np.random.rand(2),[b12,b21,b31,b32])['x']
        min_ls = np.append(min_l2,1/2 - 1/min_l2[0] - b32/min_l2[0])
        pt_on_var = vari(min_ls) #makes lambda3 = 0
        options['l2min'] = {(spe1,spe2):pt_on_var[0],(spe2,spe1):pt_on_var[1],(spe3,spe1):pt_on_var[2],(spe3,spe2):pt_on_var[3]}
    elif pair_key == ():
        min_ls = minimize(gfun,np.random.rand(3),[b12,b21,b31,b32])
        pt_on_var = vari(min_ls)
        options['l2min'] = {(spe1,spe2):pt_on_var[0],(spe2,spe1):pt_on_var[1],(spe3,spe1):pt_on_var[2],(spe3,spe2):pt_on_var[3]}
    return options

##Distance to variety l3 = 0, projected
def g12(q,b_list):
    '''Compute distance to l3=0 where l3 is the third eigenvalue of the fixed point on the sp1/sp2 plane projected onto bt12=C1,bt21=C2
    Dependency: None
    '''
    bt12,bt21,bt31,bt32 = b_list
    s = bt12 + 1
    t = bt21 + 1
    return (bt31 - (q*t - 1 + t/2))**2 + (bt32 - ( -s*q - 1 +s/2))**2

##Distance to variety l3 = 0, projected
def g13(x_list,b_list):
    '''Compute distance to l3=0 where l3 is the third eigenvalue of the fixed point on the sp1/sp2 plane projected onto bt13=C1
    Dependency: None
    '''
    bt12,bt21,bt31,bt32 = b_list
    s,t = x_list
    q = 1/t - 1/2 + bt31/t
    return (bt12 - (s-1))**2 + (bt21 - (t-1))**2 +  (bt32 - ( -s*q - 1 +s/2))**2

##Distance to variety l3 = 0, projected
def g23(x_list,b_list):
    '''Compute distance to l3=0 where l3 is the third eigenvalue of the fixed point on the sp1/sp2 plane projected onto bt23=C1
    Dependency: None
    '''
    bt12,bt21,bt31,bt32 = b_list
    s,t = x_list
    q = 1/2 - 1/s - bt32/s
    return (bt12 - (s-1))**2 + (bt21 - (t-1))**2 + (bt31 - (q*t - 1 + t/2))**2 + (bt32 - ( -s*q - 1 +s/2))**2

##Make a random perturbation of trio interaction parameters
def make_perturber(delta,pr,df):
    '''Makes a random matrix with 0s on the diagonal and at (pr[0],pr[1]) and (pr[1],pr[0])
    Dependecy:pandas,numpy'''
    randmat = delta*np.random.rand(*df.shape) - delta/2*np.ones(df.shape)
    perter = pd.DataFrame(randmat,index = df.index, columns = df.columns)
    for i in perter.index:
        perter.loc[i,i] = 0
    perter.loc[pr[0],pr[1]] = 0
    perter.loc[pr[1],pr[0]] = 0
    return perter

##Seed search with a point on the variety l3=0
def make_seed(w1,w11,scld,kvals_rep,rvals_rep,w1tris,g,pair_outs):
    '''without changing the parameters between
    some pair within the trio, make a seed for a search by finding a point with
    that makes the extinction states that need to be changed to have a central eigenvector
    w1 - the pair you want to preserve tuple
    w11 - the trio in question tuple
    scld - DataFrame of interactions
    kvals_rep
    rvals_rep
    w1tris - DataFrame of trio outcomes
    g - distance from the center variety
    pair_outs - pair outcomes
    Dependecy: pandas,numpy,make_center,
    '''
    g12,g13,g23 = g

    w11p = scld.loc[list(w11),list(w11)]#the interactions between just this trio
    w11o = w1tris.loc[[w11],'Observed'][0] #observed trio outcome string like Xx-Xx
    w11pr = w1tris.loc[[w11],'gLV Predictions'][0] #original gLV predicted outcome string like Xx-Xx
    last_guy = list(set(w11)-set(w1))[0] #the one in the trio that isn't in the pair being inspected

    new_inters = scld.loc[w11,w11]
    if w1tris.loc[[w11],'gLVRight'].values:#if it ain't broke...
        return True, new_inters

    fail = False

    #the three extinction states
    st12 = '-'.join(w1)
    st12_set = set(w1)
    st13 = '-'.join([w1[0],last_guy])
    st13_set = {w1[0],last_guy}
    st23 = '-'.join([last_guy,w1[1]])
    st23_set = {last_guy,w1[1]}


    w11o_set = set(w11o.split('-'))
    w11pr = w11pr.split(' & ')
    w11pr_set = [set(s.split('-')) for s in w11pr]

    #which ones need fixing?
    st12_fix = st12_set == w11o_set or any([wst == st12_set for wst in w11pr_set])
    st13_fix = st13_set == w11o_set or any([wst == st13_set for wst in w11pr_set])
    st23_fix = st23_set == w11o_set or any([wst == st23_set for wst in w11pr_set])


    key31 = '.'.join(['b',last_guy,w1[0]])
    key32 = '.'.join(['b',last_guy,w1[1]])
    key23 = '.'.join(['b',w1[1],last_guy])
    key13 = '.'.join(['b',w1[0],last_guy])
    #if this one does, etc
    if st12_fix:
        fix_opts = make_center(w11,st12,new_inters,g12,(1,2))
        if not (st13_fix or st23_fix):
            new_inters_d = fix_opts['l2min']
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]
        elif st13_fix:
            new_inters_d = fix_opts[key32]
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]
        elif st23_fix:
            new_inters_d = fix_opts[key31]
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]


    if st13_fix:
        fix_opts = make_center(w11,st13,new_inters,g13,(1,3))
        if not (st12_fix or st23_fix):
            new_inters_d = fix_opts['l2min']
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]
        elif st12_fix:
            new_inters_d = fix_opts[key23]
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]
        elif st23_fix:
            new_inters_d = fix_opts[key31]
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]

    if st23_fix:
        fix_opts = make_center(w11,st23,new_inters,g23,(2,3))
        if not (st12_fix or st13_fix):
            new_inters_d = fix_opts['l2min']
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]
        elif st12_fix:
            new_inters_d = fix_opts[key23]
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]
        elif st13_fix:
            new_inters_d = fix_opts[key32]
            for ky in new_inters_d.keys():
                new_inters.loc[ky] = new_inters_d[ky]

    ### Now any extinction state that needed fixing has a center manifold. If the other two eigenvalues are
    #### the wrong sign, we likely can't reconcile that with the pair data


    #### What if we need to go double extinction to co-existence or vice versa?
    ### if we need to go to double extinction we can do that
    if w11o.count('-') == 0:
        oth2 = list(set(w11) - {w11o})
        if new_inters.loc[oth2[0],w11o] > -1:
            if set([oth2[0],w11o]) == w1:
                fail = True
            else:
                new_inters.loc[oth2[0],w11o] = -1.00001
        if new_inters.loc[oth2[1],w11o] > -1:
            if set([oth2[1],w11o]) == w1:
                fail = True
            else:
                new_inters.loc[oth2[1],w11o] = -1.00001

    ### if had a double exitnction we can get rid of it...as long as that doesn't requre changing
    ### one of our pair interactions in the pair we don't want to change. However, if that's the case
    ### we might still have bistability and it's ok.
    for ws in w11pr:
        if ws.count('-') == 0 and ws != 'Blow Up':
            oth2 = list(set(w11) - {ws})
            if new_inters.loc[oth2[0],ws] < -1:
                if not set([oth2[0],ws]) == w1:
                    new_inters.loc[oth2[0],ws] = -0.99999
            if new_inters.loc[oth2[1],ws] < -1:
                if not set([oth2[1],ws]) == w1:
                    new_inters.loc[oth2[1],ws] = -0.99999

    ### Ok. All the above gives us a starting point for a search for a good parameter set. Fun!
    if fail:
        return False, new_inters
    else:
        return True, new_inters

##fit function for genetic search (single trio)
def gen_fit(params ,trio,kvals, rvals,trio_observation,pair_observation,desired_stab):
    '''Asses fitness of a parameter set - min_{lambda gives correct stability}dist(eigenvalus,lambda), unless the pair is wrong, then it's unfit
    Dependency: pandas, numpy,test_trio_LV,threesp_eq, jac_thr,'''
    all_tst,pair_tst,eigens,stabs = test_trio_LV(trio, params,kvals, rvals,trio_observation,pair_observation, details = True,allow_bistab = True, searching = True)
    if all_tst: #hey this one works!
        fit = 0
    else: #it doesn't...how close is it?
        fit = np.linalg.norm(np.array(eigens[desired_stab][0])) + 0.001
        if not pair_tst:
            fit = 100
    return fit

##Create new interaction parameters from old set (single trio)
def gen_mate(set1,set2,pr, mut_chance = 0.2):
    '''Mate two parameter sets for a genetic algorithm, and have some chance of perturbation
    Dependecy: numpy, numpy.random, pandas'''
    mom_or_dad = rand(3) < 0.5
    child = pd.DataFrame(index = set1.index,columns = set1.columns)
    for i in range(len(child)):
        if mom_or_dad[i]:
            child.iloc[i] = set1.iloc[i]
        else:
            child.iloc[i] = set2.iloc[i]
    if all(child == set1) or all(child == set2):
        mut_chance = 1
    if rand()< mut_chance:
        child = child + make_perturber(0.05, pr,child)
    return child

##Genetic algorithm search for working parameters for a single trio
def gen_search(seed,pair,trio,kvals, rvals,trio_observation,pair_observation,seed_dist = 0.2, max_mins = 2, ncores = 0):
    '''Search for a set of parameters that matches the observation without changing interaction
    between the pair.
    seed - seed parameter set (interactions), DataFrame
    pair - tuple of 'Xx'
    trio - tuple of 'Xx'
    trio_observation - outcome of trio as string 'Xx' or 'Xx-Xx' or 'Xx-Xx-Xx'
    pair_observation - outcome of the 3 pair experiments involved in the trio as DataFrame
                make: by pair_out_pf.loc[[set(a).issubset(trio) for a in pair_out_pf.index]]
    threesp_eq - the 3 species steady states (for testing stability)
    jac_thr - the 3 species jacobian matrix (for testing stability)
    Dependency: pandas,make_perturber,numpy,time,itertools,gen_fit,gen_mate,threesp_eq, jac_thr
    '''
    if ncores == 0:
        ncores = mp.cpu_count() - 2
    last_guy = list(set(trio)-set(pair))[0]
    desired_stab = pd.Series(index = ['0',trio[0],trio[1],trio[2],'-'.join([trio[0],trio[1]]),'-'.join([trio[0],trio[2]]),'-'.join([trio[1],trio[2]]),'-'.join([trio[0],trio[1],trio[2]])])
    for pt in desired_stab.index:
        desired_stab.loc[pt] = set(pt.split('-')) == set(trio_observation.split('-'))
    population = [seed + make_perturber(seed_dist,pair,seed) for i in range(10)]#initial population
    fits = [gen_fit(fndr,trio,kvals, rvals,trio_observation,pair_observation,desired_stab) for fndr in population]
    if min(fits)==0:#hurray it found it right away!
        winners_ordered = np.array(fits).argsort()
        winners = [population[i] for i in winners_ordered[:4] if fits[i] == 0]
    else:
        stp_time = time.time() + 60*max_mins
        while min(fits)>0  and time.time() < stp_time:
            winners_ordered = np.array(fits).argsort()
            winners = [population[i] for i in winners_ordered[:4] if fits[i] < 100]
            if len(winners) == 0:
                winners = [seed + make_perturber(seed_dist,pair,seed) for i in range(10)]#reset with new initial guess
            population = jb.Parallel(n_jobs = ncores)(jb.delayed(gen_mate)(win[0],win[1],pair)for win in it.combinations(winners,2)) + winners
            # population = [gen_mate(win[0],win[1],pair) for win in it.combinations(winners,2)] + winners
            fits = jb.Parallel(n_jobs = ncores)(jb.delayed(gen_fit)(fndr,trio,kvals, rvals,trio_observation,pair_observation,desired_stab) for fndr in population)
            # fits = [gen_fit(fndr,trio,kvals, rvals,trio_observation,pair_observation,desired_stab) for fndr in population]
            # if time.time() > stp_time:
            #     break
    return min(fits) == 0, winners[0]

##Make a random perturbation of all interaction parameters
def make_big_perturber(delta,df):
    '''Makes a random matrix with 0s on the diagonal
    Dependency: numpy,pandas'''
    randmat = delta*np.random.rand(*df.shape) - delta/2*np.ones(df.shape)
    perter = pd.DataFrame(randmat,index = df.index, columns = df.columns)
    for i in perter.index:
        perter.loc[i,i] = 0
    return perter

##fit function for genetic search (all trios)
def gen_big_fit(params,kvals, rvals,trio_observation,pair_observation,trio_list):
    '''Asses fitness of a parameter set - min_{lambda gives correct stability}dist(eigenvalus,lambda), unless a pair is wrong, then it's unfit
    Dependency: test_params_LV, numpy, pandas
    '''
    all_tst,pair_tst,trio_out,deets = test_params_LV(params, kvals, rvals, trio_observation, pair_observation, details = True, allow_bistab = True, searching = True)
    fit = 0
    if all_tst: #hey this one works!
        return fit
    else: #it doesn't...how close is it?
        for tri in deets.keys():
            # states = [{tri[0]}, {tri[1]],[tri[2]},{tri[0],tri[1]},{tri[1],tri[2]},{tri[0],tri[2]},{tri[0],tri[1],tri[2]}]
            # want_stable =  np.array([st == set(trio_observation.loc[tri,'Observed'][0].split('-')) for st in states])
            eigns = deets[tri].loc[trio_observation.loc[[tri],'Observed'],'eigenvalues']
            fit_t = np.linalg.norm(eigns)
            fit = fit + fit_t
        if not pair_tst:
            fit = 1000
        return fit

##Create new interaction parameters from old set (all trios)
def gen_big_mate(set1,set2, mut_chance = 0.2):
    '''Mate two parameter sets for a genetic algorithm, and have some chance of perturbation
    Dependency:numpy,pandas,numpy.random, make_big_perturber
    '''
    mom_or_dad = np.random.rand(len(set1)) < 0.5
    child = pd.DataFrame(index = set1.index,columns = set1.columns)
    for i in range(len(child)):
        if mom_or_dad[i]:
            child.iloc[i] = set1.iloc[i]
        else:
            child.iloc[i] = set2.iloc[i]
    if all(child == set1) or all(child == set2):
        mut_chance = 1
    if rand()< mut_chance:
        child = child + make_big_perturber(0.05,child)
    return child

##Genetic algorithm search for working parameters for a all trios at once
def big_gen_search(seed,kvals,rvals,trio_observation,pair_observation,seed_dist = 0.2, max_mins = 2):
    '''Search for a set of parameters that matches the observation
    seed - seed parameter set (interactions), DataFrame
    trio_observation - outcome of trios as series as string 'Xx' or 'Xx-Xx' or 'Xx-Xx-Xx'
    pair_observation - outcome of the pair experiments as series
    threesp_eq - the 3 species steady states (for testing stability)
    jac_thr - the 3 species jacobian matrix (for testing stability)
    Dependency: pandas,time,itertools,make_big_perturber,gen_big_fit,numpy,gen_big_mate,threesp_eq, jac_thr
    '''
    population = [seed + make_big_perturber(seed_dist,seed) for i in range(10)]#initial population
    trio_list = trio_observation.index
    ncores = mp.cpu_count() - 2
    fits = [gen_big_fit(fndr,kvals, rvals,trio_observation,pair_observation,trio_list) for fndr in population]
    if min(fits)==0:#hurray it found it right away!
        winners_ordered = np.array(fits).argsort()
        winners = [population[i] for i in winners_ordered[:4] if fits[i] == 0]#give 4 of them for some reason
    else:
        failcount = 1
        stp_time = time.time() + 60*max_mins
        while min(fits)>0 and time.time() < stp_time:
            winners_ordered = np.array(fits).argsort()
            winners = [population[i] for i in winners_ordered[:4] if fits[i] < 100]
            if len(winners) == 0:
                winners = [seed + make_big_perturber(3*failcount*seed_dist,seed) for i in range(10)]#reset with new initial guess
                failcount = failcount + 1
            population = jb.Parallel(n_jobs = ncores)(jb.delayed(gen_big_mate)(win[0],win[1])for win in it.combinations(winners,2)) + winners
            # population = [gen_big_mate(win[0],win[1]) for win in it.combinations(winners,2)] + winners
            fits = jb.Parallel(n_jobs = ncores)(jb.delayed(gen_big_fit)(fndr,kvals, rvals,trio_observation,pair_observation,trio_list) for fndr in population)
            # fits = [gen_big_fit(fndr,kvals, rvals,trio_observation,pair_observation,trio_list) for fndr in population]
            # if time.time() > stp_time:
            #     break
    return min(fits) == 0, winners[0]

##Use gen_search for each trio a pair is in
def fix_pair_pos_gen(pr,scld,kvals_rep,rvals_rep,pair_outs,tri_out, max_minutes = 2):
    '''Attempt to fix every trio a pair is included in without changing the pair (using gen search)
    Dependency: pandas,make_seed,gen_search, g12,g13,g23
    '''
    gfs = [g12,g13,g23]
    pr_trios = tri_out[[set(pr).issubset(ti) for ti in tri_out.index]]#find the trios, get DF
    new_ps = scld.copy()
    params = {}
    worked = {}
    for trio in pr_trios.index:
        # print(trio)
        if pr_trios.loc[[trio],'gLVRight'][0]:
            worked[trio] = True
            params[trio] = scld.loc[list(trio),list(trio)]
        else:
            pobs = pair_outs.loc[[set(a).issubset(trio) for a in pair_outs.index]]
            seeder = make_seed(pr,trio,scld,kvals_rep,rvals_rep,pr_trios,gfs,pair_outs)[1]
            param_work = gen_search(seeder,pr,trio,kvals_rep,rvals_rep,pr_trios.loc[[trio],'Observed'][0],pobs,max_mins =max_minutes/len(pr_trios))
            worked[trio] = param_work[0]
            if worked[trio] == True:
                params[trio] = param_work[1]
    return worked,params

##Use fix_pair_pos_gen for all pairs
def fix_all_pairs_gen(all_pairs, params, ks, rs, pr_outcoms, tri_outcomes, how_long_to_wait = 2):
    '''Iterating through all pairs, try to fidx the pair without changing the pair's internal parameters.
    Dependency: fix_pair_pos_gen,pandas'''
    pair_can_work = {}
    pair_working_parameters = {}
    for ppi in all_pairs.index:
        #print(ppi)
        fix_it = fix_pair_pos_gen(ppi,params,ks,rs,pr_outcoms,tri_outcomes, max_minutes = how_long_to_wait/len(all_pairs))
        pair_can_work[ppi] = fix_it[0]
        pair_working_parameters[ppi] = fix_it[1]
    return pair_can_work, pair_working_parameters

##Use big_gen_search
def find_full_sol(paramers,kvals,rvals,trio_outcomes,pair_outcomes, fix_prs, max_mints = 0.1):
    '''Try to find a full solution using big_gen_search, seeded by making 12 independed trios work
    paramers: original parameters
    trio_outcomes: outcome of trios as series as string 'Xx' or 'Xx-Xx' or 'Xx-Xx-Xx'
    pair_outcomes: outcome of the pair experiments as series
    threesp_eq: the 3 species steady states (for testing stability)
    jac_thr: the 3 species jacobian matrix (for testing stability)
    fix_prs: A list, output of fix_all_pairs_gen. First entry is dictionary of pair:proportion that were fixed. Second entry is dictionary of pair:parameters that are good
    Dependency:pandas, itertools ,threesp_eq, jac_thr
    '''
    params = paramers.copy()
    pair_list = pair_outcomes.index
    trio_list = trio_outcomes.index
    ###use the pair experiment to seed with at least some correct trios!
    suc_of_pairs = {}
    for ky in fix_prs[0].keys():
        suc_of_pairs[ky] = sum(list(it.chain(fix_prs[0][ky].values())))/len(fix_prs[0][ky].values())
    pr1 = sorted(suc_of_pairs,key=suc_of_pairs.get)[-1]
    for tr in fix_prs[1][pr1]:
        params.loc[fix_prs[1][pr1][tr].index,fix_prs[1][pr1][tr].columns] = fix_prs[1][pr1][tr]
    pr2 = sorted(suc_of_pairs,key=suc_of_pairs.get)[-2]
    for tr in fix_prs[1][pr2]:
        if not set(pr1).issubset(tr):
            params.loc[fix_prs[1][pr2][tr].index,fix_prs[1][pr2][tr].columns] = fix_prs[1][pr2][tr]
    pr3 = sorted(suc_of_pairs,key=suc_of_pairs.get)[-3]
    for tr in fix_prs[1][pr3]:
        if not (set(pr1).issubset(tr) or set(pr1).issubset(tr)):
            params.loc[fix_prs[1][pr3][tr].index,fix_prs[1][pr3][tr].columns] = fix_prs[1][pr3][tr]
    workd, new_params = big_gen_search(params, kvals,rvals,trio_outcomes,pair_outcomes,seed_dist = 0.2, max_mins = max_mints)
    if workd:
        return workd, new_params
    else:
        return workd, params


######### For estimating parameters:
##Compute value of logistic curve
def make_logistic_fun(x0):
    def logistic_fun(t,r,K):
        '''This is the solution to the logistic equations dx/dt = rx(1-x).
        Dependency: numpy'''
        return x0*K/(x0+(K-x0)*np.exp(-r*t))
    return logistic_fun



##Fit to logistic curve
def logistic_fit(data_df, a = 0.01):
    '''fit a solution to the logistic equation to the data_df, removing the first data point for a better fit.
    Dependency: numpy, pandas, scipy.optimize'''
    x = np.meshgrid(np.arange(data_df.shape[0]),np.arange(data_df.shape[1]))[0].ravel()
    y_data = np.transpose(data_df.values).ravel()
    rmv_pts = np.isnan(y_data)
    y_data = y_data[~rmv_pts]
    x = x[~rmv_pts]
    rmv_start = x == 0
    y_data = y_data[~rmv_start]
    x = x[~rmv_start]
    x1s = y_data[x==1]
    xzero = np.mean(x1s)
    logi_f = curve_fit(make_logistic_fun(xzero),x,y_data,bounds=([ 0,a], [np.inf, np.inf]))
    r = logi_f[0][0]
    k = logi_f[0][1]
    return [r,k]

##Fit to logistic curve
def logistic_fit_nolag(data_df, a= 0.01):
    '''fit a solution to the logistic equation to the data_df.
    Dependency: numpy, pandas, scipy.optimize'''
    x = np.meshgrid(np.arange(data_df.shape[0]),np.arange(data_df.shape[1]))[0].ravel()
    y_data = np.transpose(data_df.values).ravel()
    rmv_pts = np.isnan(y_data)
    y_data = y_data[~rmv_pts]
    x = x[~rmv_pts]
    xzero = np.mean(y_data[x == 0])
    logi_f = curve_fit(make_logistic_fun(xzero),x,y_data,bounds=([0, a], [np.inf, np.inf]))
    r = logi_f[0][0]
    k = logi_f[0][1]
    return [r,k]

##Residules for model fit
def model_traj_res(a_vals,r_vals,data,stime,printer = False):
    '''
    For the 2 species model!
    Compute the residules sol(t) - data(t), where t = [0,1,2,3,4,5]*num_experiments. Pass these
    residules to least_squares to fit the parameters.
    a_vals - optimize over these
    r_vals - list [r1,r2] of growth rates
    data - pd.DataFrame multindexed by experiment number & time. 2 columns - one for each species
    Dependency: pandas,matplotlib,solve_vode,numpy, twospec_num
    '''
    #figure out the x0s...are there 2????
    exps_ind = data.index.levels[0] #list of what the experiments are indexed as
    init_condits = [data.loc[i,0.0].values for i in data.index.levels[0]]#initial conditions of experiments
    all_resids = []
    if printer:
        fig,axs = plt.subplots(2,1)
        inits1 = data.loc[1].loc[0,:].values
    for exp in exps_ind:
        if printer:
            if all(data.loc[exp].loc[0,:].values == inits1):
                ax = axs[0]
            else:
                ax = axs[1]
        x0 = data.loc[exp].loc[stime,:].values
        sol,sol_times = solve_vode(twospec_num,[r_vals,a_vals],x0,5,dt = 1,t0 = stime)
        exp_data_df = data.loc[exp].iloc[1:]
        sol_care = sol[[np.float64(ti) in exp_data_df.index for ti in sol_times]]
        exp_data = exp_data_df.values
        if printer:
            for p in range(len(sol_care)):
                ax.plot([sol_care[p,0],exp_data[p,0]],[sol_care[p,1],exp_data[p,1]], marker = 'o', color = 'b')
            ax.plot(sol_care[:,0],sol_care[:,1], color = 'r')
        if len(sol_care) < len(exp_data):#we have finite time blow up so the integrator stopped.
            more_pts = len(exp_data) - len(sol_care)
            sol_care = np.pad(sol_care,((0,more_pts),(0,0)),'constant', constant_values = 10**10)
        diffss = np.sum((exp_data - sol_care)**2, axis = 1)**0.5
        all_resids = all_resids + list(diffss)
    all_resids = np.array(all_resids).ravel()
    if np.prod(a_vals)>1:
        return np.array([1000*np.prod(a_vals)]*len(all_resids))
    else:
        return all_resids

#fit 2 species model
def model_fit_wrapper(pair_df, rvals, stime = 1, numtris = 1,masterbound = np.inf):
    '''Parameter fitting using least_squares
    Dependency: pandas, numpy.random, scipy.optimize, model_traj_res'''
    ##First, let's cheat the parameters a little bit by deciding ourselves if we should have an
    ## extinction event.
    exps_ind = pair_df.index.levels[0] #list of what the experiments are indexed as
    exps_lasts = [pair_df.loc[exmt].index[-1] for exmt in exps_ind]#index of final values of experiment
    end_of_exps = [pair_df.loc[pd.IndexSlice[exps_ind[i],exps_lasts[i]],:].values for i in range(len(exps_ind))]#final exp values
    extinction_1 = all([False if x[1] == 0 else x[0]/x[1] < 0.1 for x in end_of_exps])#does species 1 go extinct
    extinction_2 = all([False if x[0] == 0 else x[1]/x[0] < 0.1 for x in end_of_exps])#does species 2 go extinct
    ### but where do we draw the extinction line??? Maybe use parameter fit to guess?
    ### idea: for any pair with ratio < 0.1, do both constrained and unconstrained and
    ### compare. If unconstrained indicates extinction, then we probably have extinction.
    ### Is this going to make my runtime a lot longer? We shall find out. Or...maybe I do that
    ### for all pairs, because we could have a slow extinction I guess. Then check who fits best
    mtd = 'trf'
    if extinction_1:#if we have species 1 extinction
        a0 = [rand() - 2.1, 2*rand() - 0.9]#[-1.5,-0.5]
        fit_ext = least_squares(model_traj_res,a0,args = (rvals,pair_df,stime), bounds = ([-masterbound,-0.999],[-1.001,masterbound]), method = mtd)
        for i in range(numtris):
            a0 = [rand() - 2.1, 2*rand() - 0.9]#[-1.5,-0.5]
            fit_ext_t = least_squares(model_traj_res,a0,args = (rvals,pair_df,stime), bounds = ([-masterbound,-0.999],[-1.001,masterbound]), method = mtd)
            if fit_ext_t.cost < fit_ext.cost:
                fit_ext = fit_ext_t
    elif extinction_2:#if we have species 2 extinction
        a0 = [2*rand() - 0.9,rand()-2.1]
        fit_ext = least_squares(model_traj_res,a0,args = (rvals,pair_df,stime), bounds = ([-0.999,-masterbound],[masterbound,-1.001]),  method = mtd)
        for i in range(numtris):
            a0 = [2*rand() - 0.9,rand()-2.1]
            fit_ext_t = least_squares(model_traj_res,a0,args = (rvals,pair_df,stime), bounds = ([-0.999,-masterbound],[masterbound,-1.001]),  method = mtd)
            if fit_ext_t.cost < fit_ext.cost:
                fit_ext = fit_ext_t
    else:#if we have no extinction
        a0 = [2*rand() - 0.9,2*rand() - 0.9]
        fit_ext = least_squares(model_traj_res,a0,args = (rvals,pair_df,stime), bounds = ([-0.999,-0.999],[masterbound,masterbound]), method = mtd)
        for i in range(numtris):
            a0 = [2*rand() - 0.9,2*rand() - 0.9]
            fit_ext_t = least_squares(model_traj_res,a0,args = (rvals,pair_df,stime), bounds = ([-0.999,-0.999],[masterbound,masterbound]), method = mtd)
            if fit_ext_t.cost < fit_ext.cost:
                fit_ext = fit_ext_t
    a0 = [3*rand() - 1.5,3*rand() - 1.5]
    fit_uncont = least_squares(model_traj_res,a0,args = (rvals,pair_df,stime), method = mtd,bounds = ([-masterbound,-masterbound],[masterbound,masterbound]))#unconstrained fit
    for i in range(numtris):
        a0 = [3*rand() - 1.5,3*rand() - 1.5]
        fit_uncont_t = least_squares(model_traj_res,a0,args = (rvals,pair_df,stime), method = mtd,bounds = ([-masterbound,-masterbound],[masterbound,masterbound]))#unconstrained fit
        if fit_uncont_t.cost < fit_uncont.cost:
            fit_uncon = fit_uncont_t
    fit_extx = fit_ext.x.round(3)
    fit_uncontx = fit_uncont.x.round(3)
    #is there a difference in the constrained and unconstrained?
    if all((fit_extx < -1) == (fit_uncontx < -1)):
        fit = fit_uncontx
    else:
        if extinction_1:
            if ((fit_uncontx < -1)[1] == True) or fit_ext.cost/fit_uncont.cost > 0.5: #all([False if x[1] == 0 else x[0]/x[1] < 0.02 for x in end_of_exps]):
                fit = fit_extx
            else:
                fit = fit_uncontx
                extinction_1 = False
        elif extinction_2:
            if ((fit_uncontx < -1)[0] == True) or fit_ext.cost/fit_uncont.cost > 0.5:# all([False if x[0] == 0 else x[1]/x[0] < 0.02 for x in end_of_exps]):
                fit = fit_extx
            else:
                fit = fit_uncontx
                extinction_2 = False
        else:#the paramters show extinction but there isnt...??
            fit = fit_extx
    return fit, np.array([extinction_1,extinction_2])



##Monte Carlo for the probability of the "fixed points" of a pair, using tau leaping & parallelization
def extinction_prob_pair_tl(pair,inter_params,rvals,x0 ='ran',endt=5,maxsims = 100,vol = 100,lftover=2):
    '''For a pair, use MC simulation to find the probability that one pair member would go extinct, using approximate simulation
    Dependency: numpy, make_k_pair,numpy.random,tau_postleap_anderson,joblib,multiprocessing'''
    two_spec_sources = np.array([[1,0],[2,0],[0,1],[0,2],[1,1],[1,1],[1,1],[1,1]])
    two_spec_prodcts = np.array([[2,0],[1,0],[0,2],[0,1],[2,1],[0,1],[1,2],[1,0]])
    ks = make_k_pair(inter_params,rvals, pair)
    pair_ar = np.array(pair)
    ncpus = mp.cpu_count()
    numsims = 0
    minvar = 1
    at_a_time = 100
    resuls ={'ded':0,pair_ar[0]:0,pair_ar[1]:0,'-'.join(pair_ar):0}
    possibils ={'ded':[0,0],pair_ar[0]:[1,0],pair_ar[1]:[0,1],'-'.join(pair_ar):[1,1]}
    while numsims <= maxsims and minvar > 0.01:
        all_sims = np.array(jb.Parallel(n_jobs = ncpus-lftover)(jb.delayed(tau_postleap_anderson)(two_spec_sources,two_spec_prodcts,ks,rand(2),endt,N = vol) for i in range(at_a_time)))
        for outc in resuls.keys():
            resuls[outc] += np.sum(np.all(all_sims.astype('bool') == possibils[outc], axis = 1))
        variances = {}
        numsims += at_a_time
        for outc in resuls.keys():
            meann = resuls[outc]/numsims
            sumsqrs = resuls[outc]*(2*meann-1) + numsims*(1-meann)**2
            variances[outc] = sumsqrs/(numsims - 1)
        minvar = 1.96*max(variances.values())/np.sqrt(numsims)
    resuls.update((x,y/numsims) for x,y in resuls.items())
    return resuls, minvar




##Monte Carlo for the probability of the "fixed points" of a trio, using tau leaping & parallelization
def extinction_prob_trio_tl(trio,inter_params,rvals,x0 ='ran',endt=5,maxsims = 100,vol = 100,lftover=2):
    '''For a trio, use MC simulation to find the probability that one pair member would go extinct using approximate simulation
    Dependency: numpy, make_k_trio,numpy.random,tau_postleap_anderson,joblib,multiprocessing'''
    three_spec_sources = np.array([[1,0,0],[2,0,0],[0,1,0],[0,2,0],[0,0,1],[0,0,2],[1,1,0],[1,1,0],[1,0,1],[1,0,1],[1,1,0],[1,1,0],[0,1,1],[0,1,1],[1,0,1],[1,0,1],[0,1,1],[0,1,1]])
    three_spec_prodcts = np.array([[2,0,0],[1,0,0],[0,2,0],[0,1,0],[0,0,2],[0,0,1],[2,1,0],[0,1,0],[2,0,1],[0,0,1],[1,2,0],[1,0,0],[0,2,1],[0,0,1],[1,0,2],[1,0,0],[0,1,2],[0,1,0]])
    ks = make_k_trio(inter_params,rvals, trio)
    trio_ar = np.array(trio)
    ncpus = mp.cpu_count()
    numsims = 0
    minvar = 1
    at_a_time = 100
    resuls = {'ded':0,trio_ar[0]:0,trio_ar[1]:0,trio_ar[2]:0,'-'.join(trio_ar[[0,1]]):0,'-'.join(trio_ar[[0,2]]):0,'-'.join(trio_ar[[1,2]]):0,'-'.join(trio_ar):0}
    possibils = {'ded':[0,0,0],trio_ar[0]:[1,0,0],trio_ar[1]:[0,1,0],trio_ar[2]:[0,0,1],'-'.join(trio_ar[[0,1]]):[1,1,0],'-'.join(trio_ar[[0,2]]):[1,0,1],'-'.join(trio_ar[[1,2]]):[0,1,1],'-'.join(trio_ar):[1,1,1]}
    while numsims <= maxsims and minvar > 0.01:
        all_sims = np.array(jb.Parallel(n_jobs = ncpus-lftover)(jb.delayed(tau_postleap_anderson)(three_spec_sources,three_spec_prodcts,ks,rand(3),endt,N = vol) for i in range(at_a_time)))
        for outc in resuls.keys():
            resuls[outc] += np.sum(np.all(all_sims.astype('bool') == possibils[outc], axis = 1))
        variances = {}
        numsims += at_a_time
        for outc in resuls.keys():
            meann = resuls[outc]/numsims
            sumsqrs = resuls[outc]*(2*meann-1) + numsims*(1-meann)**2
            variances[outc] = sumsqrs/(numsims - 1)
        minvar = 1.96*max(variances.values())/np.sqrt(numsims)
    resuls.update((x,y/numsims) for x,y in resuls.items())
    return resuls, minvar
############# Miscellaneous
def rescale_model(interacts,kvals):
    '''Rescale model so carrying capacities are all 1
    Dependency:pandas, numpy'''
    return -interacts.copy()*np.outer(1/kvals,kvals)

def unscale_model(interacts2,kvals):
    '''Rescale model so carrying capacities are biological.
    Dependency:pandas, numpy'''
    return -interacts2.copy()*np.outer(kvals,1/kvals)
