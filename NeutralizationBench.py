#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 04:41:01 2020

@author: Sina

A suite to simulate the reaction between any number of antigens and antibodies
assuming each antibody targets only a single site on each antigen.

"""


def _systemOfEquations(number_of_antigens,number_of_antibodies,forward_rates,backward_ratios,interference_matrix,print_equations=False):
    
    """
    Given number_of_antigens,number_of_antibodies,forward_rates,backward_ratios
    and interference_matrix it returns a functions which represents
    the system of equations for a system of the form
    
    V_kAb_I + Ab_j <=> V_kAb_{I union i}
    
    where V_k is the kth antigen, Ab_I is the collection of antibodies for
    set of integers I and Ab_j is the jth antibody and V_kAb_{I union i} is the
    complex formed by V_k and Ab_{I union i}
    
    forward rates determine the forward reaction rate of each equation
    
    V_i + Ab_j => V_iAbj
    
    backward rates are determined by multiplying forward rates with backward ratios
    
    interference_matrix[i,j,k] determines how the binding of antibody j interferes
    with binding of antibody k for antigen i. If it is 1, there is no interference
    if it is 0 then it is full interference.
    
    """
    
    from sympy import symbols,lambdify
    from itertools import combinations
    import re
  
    
    if number_of_antigens==0 or number_of_antibodies==0:
        return None,None
    
    s1,s2=forward_rates.shape
    s3,s4=backward_ratios.shape
    s5,s6,s7=interference_matrix.shape
    
    if s1!=number_of_antigens or s2!=number_of_antibodies:
        raise ValueError('Forward rates must have shape number_of_antigens x number_of_antibodies. It is {} vs ({},{})'.format(forward_rates.shape,number_of_antigens,number_of_antibodies))
    
    if s3!=s1 or s4!=s2:
        raise ValueError('Backward ratios must have the same shape as forward rates but they are {} and {}'.format(forward_rates.shape,backward_ratios.shape))
    
    if s5!=number_of_antigens or s6!= number_of_antibodies or s7!= number_of_antibodies:
        raise ValueError('Interference matrix must have shape number_of_antigens x number_of_antibodies x number_of_antibodies but it was {}.'.format(interference_matrix.shape))
    
    
    
    variables=[] #non antibody variables
    all_variables=[] #all variables: antigen,antibody,and complexes of them
    
    V=symbols('V0:'+str(number_of_antigens)) #antigen symbolic variables
    Ab=symbols('Ab0:'+str(number_of_antibodies)) #antibody symbolic variables
    
    VAb={} #dictionary mapping names in string format like V0 to its corresponding symbol
    odes={} #dictionary as above but mapping it to the variables symbolic ode
    ode_variable_indices={} #a dictionary as above mapping to the equations where these variables appear

    symbolic_vars=[] #list of symbolic variables in the system
    
    for i in range(number_of_antigens):
        antigen_name='V'+str(i)
        VAb[antigen_name]=V[i]
        symbolic_vars.append(V[i])
        variables.append(antigen_name)    
        all_variables.append(antigen_name)   
        odes[antigen_name]=0
        ode_variable_indices[antigen_name]=[]
        
    for i in range(number_of_antibodies):
        sera_name='Ab'+str(i)
        odes[sera_name]=0
        symbolic_vars.append(Ab[i])
        ode_variable_indices[sera_name]=[]
        all_variables.append(sera_name)
    
    
    bound_antibodies={}
    for i in range(number_of_antigens):
        antigen_name='V'+str(i)
        for j in range(1,number_of_antibodies+1):
            
            
            indices=combinations(range(number_of_antibodies),j)
            
            for x in indices:
                lx=list(x)
                sx=''.join([str(z) for z in lx])
                compound_name='Ab'+sx
                VAb[antigen_name+compound_name]=symbols(antigen_name+compound_name)
                bound_antibodies[antigen_name+compound_name]=lx
                odes[antigen_name+compound_name]=0
                ode_variable_indices[antigen_name+compound_name]=[]
                variables.append(antigen_name+compound_name)
                all_variables.append(antigen_name+compound_name)  
                symbolic_vars.append(VAb[antigen_name+compound_name])
                
    #odes are ordered as V 0:n, Ab 0:m,VAb 0:k          
    RHS=[] #list of variables on the righthand sides for equations, each entry corresponds to one equation
    LHS=[] #same as above but lefthand side
    ks=[] #forward and backward constants for the equations
    
    for var in variables:
        antigen_name=re.findall('V\d+',var)[0]
        i=int(antigen_name[1:])
        if len(re.findall('V\d+$',var))>0:
            for j in range(number_of_antibodies):
                k=forward_rates[i,j]
                b=backward_ratios[i,j]*k
                sera_name='Ab'+str(j)
                LHS.append((var,sera_name))
                RHS.append(var+sera_name)
                ks.append((k,b))
        else:
            bounds=bound_antibodies[var]
            for j in range(number_of_antibodies):
                if j not in bounds:
                    k=forward_rates[i,j]
                    b=backward_ratios[i,j]*k
                    interference=1
                    for l in bounds:
                        interference=interference*interference_matrix[i,l,j] #interference on j due to l being bound to virus j
                    k=k*interference
                    
                    sera_name='Ab'+str(j)
                    LHS.append((var,sera_name))
                    new_bound=sorted(bounds+[j])
                    compound_name='Ab'+''.join([str(x) for x in new_bound])
                    RHS.append(antigen_name+compound_name)
                    ks.append((k,b))
                    
    for lterms,rterm,k in zip(LHS,RHS,ks):
         lterm1=lterms[0]
         lterm2=lterms[1]
         ai=int(lterm2[2:])
         f=k[0]
         b=k[1]
         
         odes[lterm1] =odes[lterm1] - f*VAb[lterm1]*Ab[ai] + b*VAb[rterm]
         odes[lterm2] =odes[lterm2] - f*VAb[lterm1]*Ab[ai] + b*VAb[rterm]
         odes[rterm]  =odes[rterm]  + f*VAb[lterm1]*Ab[ai] - b*VAb[rterm]
         
         ode_variable_indices[lterm1]=sorted(list(set(ode_variable_indices[lterm1]+[all_variables.index(lterm1),all_variables.index(lterm2),all_variables.index(rterm)])))
         ode_variable_indices[lterm2]=sorted(list(set(ode_variable_indices[lterm2]+[all_variables.index(lterm1),all_variables.index(lterm2),all_variables.index(rterm)])))
         ode_variable_indices[rterm]=sorted(list(set(ode_variable_indices[rterm]+[all_variables.index(lterm1),all_variables.index(lterm2),all_variables.index(rterm)])))

    ode_funs=[]
    for var in all_variables:
        ode=odes[var]
        ode_funs.append(lambdify([symbolic_vars[i] for i in ode_variable_indices[var]],ode,'numpy'))
        
    if print_equations:
        print('System of equations is:\n')       
        for i,var in enumerate(all_variables):
            print(str(i)+'- d['+str(var)+']/dt=',end='')
            print(odes[var])
            print('')
        
    def lfun(t,z,ode_funs,all_variables,variable_indices):
    
        y=[]
        
        
        for fun,var in zip(ode_funs,all_variables):
            val=[z[i] for i in ode_variable_indices[var]]
            
            y.append(fun(*val))
        
        
        return y
        
       
    ode_fun= lambda t,z : lfun(t,z,ode_funs,all_variables,ode_variable_indices)
  
        
        
    return ode_fun,all_variables




def titrateAntigensAgainstSera(init_vals,dilutions,number_of_antigens,number_of_antibodies,measurement_time,forward_rates,backward_ratios,interference_matrix,print_equations=False):
    
    """
    For n antigens and m antibodies, given init_vals in order for the n antigens
    and m antibodies, the forward rates, backward ratios, interference_matrix
    and a set of dilutions, this function computes the titer curve of each
    antigen for the mixture involving these antibodies.     
    
    The system of equations for this reaction is given by:
    
    V_kAb_I + Ab_j <=> V_kAb_{I union i}
    
    where V_k is the kth antigen, Ab_I is the collection of antibodies for
    set of integers I and Ab_j is the jth antibody and V_kAb_{I union i} is the
    complex formed by V_k and Ab_{I union i}
    
    forward rates determine the forward reaction rate of each equation
    
    V_i + Ab_j => V_iAbj
    
    backward rates are determined by multiplying forward rates with backward ratios
    
    interference_matrix[i,j,k] determines how the binding of antibody j interferes
    with binding of antibody k for antigen i. If it is 1, there is no interference
    if it is 0 then it is full interference.
    
    """
    
    
    import numpy as np
    from scipy.integrate import solve_ivp

    init_vals=[float(x) for x in init_vals]
    
    #create the sustem of equations used to describe the system
    ode_fun,all_variables=_systemOfEquations(number_of_antigens,number_of_antibodies,forward_rates,backward_ratios,interference_matrix,print_equations=print_equations)                
    number_of_variables=len(all_variables)

    y3=[0]*(number_of_variables-len(init_vals))
    
    init_vals=np.array(list(init_vals)+list(y3))

    tspan=[0,measurement_time]
    y=np.zeros((number_of_antigens,len(dilutions)))
    
    nondiluted_sera_vals =init_vals[number_of_antigens:number_of_antibodies+1].copy()
    for i,dil in enumerate(dilutions):
        init_vals[number_of_antigens:number_of_antibodies+1]=nondiluted_sera_vals*dil
        sol = solve_ivp(ode_fun, tspan,init_vals)
        
        for j in range(number_of_antigens):
            y[j,i]=(sol.y[j][-1]/init_vals[j])
    
    
    log_titers=[]
    titers=[]
    log2_dilutions=[-np.log2(1/x) for x in dilutions ]
    for j in range(number_of_antigens):
        try:
            a0=np.argwhere(y[j,:]>=0.5)[-1][0]
        except:
            titers.append('>'+str(int(1/dilutions[0])))
            log_titers.append(-np.log2(1/dilutions[0]))
            continue
        try:
            a1=np.argwhere(y[j,:]<0.5)[0][0]
        except:
            titers.append('<'+str(int(1/dilutions[-1])))
            log_titers.append(-np.log2(1/dilutions[-1]))
            continue
        val0=y[j,a0]
        val1=y[j,a1]
        
        dil0=log2_dilutions[a0]
        dil1=log2_dilutions[a1]
        
        titer=dil1 + (dil0-dil1)*(0.5-val1)/(val0-val1)
        log_titers.append(titer)
        titers.append(int(2**(-titer)))
    return y,log_titers,titers,sol.y


def outputTiterCurvePlot(y,titers,number_of_antigens,dilutions,strain_names=None,fig=None,ax=None):
    
    """
    Print titer curve for each virus
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.style.use('ggplot')


    if strain_names is not None:        
        try:
            if len(strain_names)!=number_of_antigens:
                raise ValueError('strain_names should be a list of names with number of elements equal to number_of_antigens')
        except:
            raise ValueError('strain_names should be a list of names with number_of_antigens elements')

    if fig is None or ax is None:
    
        fig,ax=plt.subplots()
    
    plot_dilutions=[-np.log2(1/x) for x in dilutions]
    for i in range(number_of_antigens):
        
        
        ax.plot(plot_dilutions,y[i,:],marker='o')
        ax.set_xticks(plot_dilutions)
        ax.set_xticklabels([1/x for x in dilutions],rotation=90)
        
        
        titer=titers[i]
            
        ax.scatter(titer,0,marker='+',color='black')
        ax.set_ylim([-0.1,1])
        
        if strain_names != None:
            ax.set_title(strain_names[i])
        
        
    return ax,fig


def outputScatterPlot(data_x,data_y,xlabel,ylabel,xticks=None,xlabels=None,yticks=None,ylabels=None):
    
    """
    Scatter plot of two variables which can be used for comparing things like
    forward_rates vs titers obtained (see k_vs_titer_notexcess as an example)
    """
    
    import matplotlib.pyplot as plt
    
    plt.style.use('ggplot')
    ss=[19.2,10.2]

    
    fig,ax=plt.subplots(figsize=(ss[0],ss[1]))
    

        
    ax.scatter(data_x,data_y)
    
    if xticks != None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels,rotation=90)
    
    if yticks != None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)    
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
        
        
    return ax,fig


