#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 04:41:01 2020

@author: Sina

A suite to simulate the reaction between any number of antigens and antibodies
assuming each antibody targets only a single site on each antigen.

"""

def _systemOfEquations(number_of_antigens, number_of_antibodies, association_rates,
                       dissociation_rates, interference_matrix,
                       assay_sensitivity=None, print_equations=False):

    """
    Given number_of_antigens,number_of_antibodies,association and dissociaton rates
    and interference_matrix it returns a functions which represents
    the system of equations for a system of the form

    V_kAb_I + Ab_j <=> V_kAb_{I union i}

    where V_k is the kth antigen, Ab_I is the collection of antibodies for
    set of integers I and Ab_j is the jth antibody and V_kAb_{I union i} is the
    complex formed by V_k and Ab_{I union i}

    association rates determine the forward reaction rate of each equation

    V_i + Ab_j => V_iAbj

    dissociation rates determine the rate of backwards reaction.

    interference_matrix[i,j,k] determines how the binding of antibody j
    interferes with binding of antibody k for antigen i. If it is 1, there is
    no interference if it is 0 then it is full interference.

    """
    import numpy as np
    from sympy import symbols, lambdify
    from itertools import combinations
    import re

    if number_of_antigens == 0 or number_of_antibodies == 0:
        return None, None, None

    if assay_sensitivity is None:
        assay_sensitivity = np.ones((number_of_antibodies,))
    elif assay_sensitivity.size != number_of_antibodies:
        raise ValueError(
            """Assay sensitivity must have number of elements equal to 
            number of antibodies""")
            
    s1, s2 = association_rates.shape
    s3, s4 = dissociation_rates.shape
    s5, s6, s7 = interference_matrix.shape

    if s1 != number_of_antigens or s2 != number_of_antibodies:
        raise ValueError("""Association rates must have shape number_of_antigens
                         x number_of_antibodies. It is {} vs ({},{})""".format
                         (association_rates.shape, number_of_antigens,
                          number_of_antibodies))

    if s3 != number_of_antigens or s4 != number_of_antibodies:
        raise ValueError("""Dissociation rates must have shape number_of_antigens
                         x number_of_antibodies. It is {} vs ({},{})""".format(
                         dissociation_rates.shape, number_of_antigens,
                         number_of_antibodies))

    if (s5 != number_of_antigens or s6 != number_of_antibodies
            or s7 != number_of_antibodies):

        raise ValueError(
            """Interference matrix must have shape number_of_antigens x 
            number_of_antibodies x number_of_antibodies but it was {}."""
            .format(interference_matrix.shape))

    variables = []  # non antibody variables
    all_variables = []  # all variables: antigen,antibody,and complexes of them

    V = symbols('V0:' + str(number_of_antigens))  # antigen symbolic variables
    Ab = symbols('Ab_0:' + str(number_of_antibodies))  # antibody symbolic variables

    VAb = {}  # dictionary mapping names in string format like V0 to its corresponding symbol
    odes = {}  # dictionary as above but mapping it to the variables symbolic ode
    ode_variable_indices = {}  # a dictionary as above mapping to the equations where these variables appear

    symbolic_vars = []  # list of symbolic variables in the system

    for i in range(number_of_antigens):
        antigen_name = 'V' + str(i)
        VAb[antigen_name] = V[i]
        symbolic_vars.append(V[i])
        variables.append(antigen_name)
        all_variables.append(antigen_name)
        odes[antigen_name] = 0
        ode_variable_indices[antigen_name] = []

    for i in range(number_of_antibodies):
        sera_name = 'Ab_' + str(i)
        odes[sera_name] = 0
        symbolic_vars.append(Ab[i])
        ode_variable_indices[sera_name] = []
        all_variables.append(sera_name)

    bound_antibodies = {}  # dictionary mapping a variable to the list of bound antbodies it contains, ex: V_i * Ab_j_k => j,k
    # here we generate all the symbolic variables associated to the system

    for i in range(number_of_antigens):
        antigen_name = 'V' + str(i)
        for j in range(1, number_of_antibodies + 1):

            indices = combinations(range(number_of_antibodies), j)

            for x in indices:
                lx = list(x)
                sx = ''.join(['_' + str(z) for z in lx])
                compound_name = 'Ab' + sx
                VAb[antigen_name + compound_name] = symbols(antigen_name +
                                                            compound_name)
                bound_antibodies[antigen_name + compound_name] = lx
                odes[antigen_name + compound_name] = 0
                ode_variable_indices[antigen_name + compound_name] = []
                variables.append(antigen_name + compound_name)
                all_variables.append(antigen_name + compound_name)
                symbolic_vars.append(VAb[antigen_name + compound_name])

    # equations are of the form V_i Ab_k + Ab_j <=> V_i Ab_{i,j}
    # in the loop below we construct lefthand sides and righthand sides
    # for these equations as well as association and dissociation rates (ks)
                
    RHS = []  # list of variables on the righthand sides for equations, each entry corresponds to one equation
    LHS = []  # same as above but lefthand side 
    ks = []   # association and dissociation rates for each equation
        
    for var in variables:
        antigen_name = re.findall(r'V\d+', var)[0]
        i = int(antigen_name[1:])
        if len(re.findall(r'V\d+$', var)) > 0:  # if the variable is only an antigen
            for j in range(number_of_antibodies):  # for each antibody add an equation V_i + A_j => V_i * A_j
                k = association_rates[i, j]
                b = dissociation_rates[i, j] 
                sera_name = 'Ab_' + str(j)
                LHS.append((var, sera_name))
                RHS.append(var + sera_name)
                ks.append((k, b))
        else:  # if the variable is a complex of the form V_i*Ab_i_j_... find the equations that add the remaining variables
            bounds = bound_antibodies[var]  # antibodies bound to this complex
            
            for j in range(number_of_antibodies):
                
                if j not in bounds:
                    k = association_rates[i, j]
                    b = dissociation_rates[i, j] 
                    sera_name = 'Ab_' + str(j)
                    interference = 1
                    for l in bounds:
                        interference = interference * interference_matrix[i, l, j]
                        # interference on j due to l being bound to virus j
                    k = k * interference

                    sera_name = 'Ab_' + str(j)
                    LHS.append((var, sera_name))
                    new_bound = sorted(bounds + [j])
                    compound_name = 'Ab' + ''.join(['_' + str(x) for x in new_bound])
                    RHS.append(antigen_name + compound_name)
                    ks.append((k, b))

    for lterms, rterm, k in zip(LHS, RHS, ks):
        lterm1 = lterms[0]
        lterm2 = lterms[1]
        ai = int(lterm2[3:])
        f = k[0]
        b = k[1]

        odes[lterm1] = odes[lterm1] - f * VAb[lterm1] * Ab[ai] + b * VAb[rterm]
        odes[lterm2] = odes[lterm2] - f * VAb[lterm1] * Ab[ai] + b * VAb[rterm]
        odes[rterm] = odes[rterm] + f * VAb[lterm1] * Ab[ai] - b * VAb[rterm]

        ode_variable_indices[lterm1] = sorted(list(set(ode_variable_indices[lterm1] + [all_variables.index(lterm1), all_variables.index(lterm2), all_variables.index(rterm)])))
        ode_variable_indices[lterm2] = sorted(list(set(ode_variable_indices[lterm2] + [all_variables.index(lterm1), all_variables.index(lterm2), all_variables.index(rterm)])))
        ode_variable_indices[rterm] = sorted(list(set(ode_variable_indices[rterm] + [all_variables.index(lterm1), all_variables.index(lterm2), all_variables.index(rterm)])))

    ode_funs = []
    inactive_proportions = []
    
    for var in all_variables:
        ode = odes[var]
        ode_funs.append(lambdify([symbolic_vars[i]
                                  for i in ode_variable_indices[var]],
                                 ode, 'numpy'))
        if len(re.findall(r'V\d+$', var)) == 1:
            inactive_proportions.append(0)
            
        elif len(re.findall(r'V\d+', var)) == 0:
            inactive_proportions.append(0)

        else: 
            bounds = bound_antibodies[var]  # antibodies bound to this complex
            ip = 0
            for j in bounds:
                ip = np.max([ip, assay_sensitivity[j]])
                
            inactive_proportions.append(ip)    
                
    if print_equations:
        print('System of equations is:\n')
        for i, var in enumerate(all_variables):
            print(str(i) + '- d[' + str(var) + ']/dt=', end='')
            print(odes[var])
            print('')

    def lfun(t, z, ode_funs, all_variables, variable_indices):

        y = []
        for fun, var in zip(ode_funs, all_variables):
            val = [z[i] for i in ode_variable_indices[var]]
            y.append(fun(*val))

        return y

    def ode_fun(t, z): 
        return lfun(t, z, ode_funs, all_variables, ode_variable_indices)  

    return ode_fun, all_variables, inactive_proportions


def titrateAntigensAgainstSera(init_vals, dilutions, number_of_antigens,
                               number_of_antibodies, measurement_time,
                               association_rates, dissociation_rates,
                               interference_matrix, assay_sensitivity=None,
                               print_equations=False):

    """
    For n antigens and m antibodies, given init_vals in order for the n
    antigens and m antibodies, the association and dissociation rates,
    interference_matrix and a set of dilutions, this function computes the
    titer curve of each antigen for the mixture involving these antibodies.

    The system of equations for this reaction is given by:

    V_kAb_I + Ab_j <=> V_kAb_{I union i}

    where V_k is the kth antigen, Ab_I is the collection of antibodies for
    set of integers I and Ab_j is the jth antibody and V_kAb_{I union i} is the
    complex formed by V_k and Ab_{I union i}

    association rates determine the forward reaction rate of each equation

    V_i + Ab_j => V_iAbj

    and dissociation rates determine the reverse reaction rate.

    interference_matrix[i,j,k] determines how the binding of antibody j
    interferes with binding of antibody k for antigen i. If it is 1, there is
    no interference if it is 0 then it is full interference.
    
    Assay sensitivity measures the sensitivity of an assay for a given 
    antibody. This is used in _systemOfEquations to compute inactive_
    proportions. If for instance assay_sensitivity[0]=0 then any virus that is
    only bound to first antibody is not inactivated. In this case all viruses
    of the form VA0 are still active. An example is if you are doing an HI
    assay, NA antibodies, even though could bind to the virus will not 
    inactivate it from the perspective of an HI assay.
    
    More generally if we denote sensitivity for i^th antibody as s_i then
    given the complex V_j Ab_{i_1} Ab_{i_2} ... Ab_{i_k}, only max_i[s_i]
    proportion of this will be inactivated. This value is stored in inactive_
    proportions which is returned from _systemOfEquations


    """

    import numpy as np
    import re
    from scipy.integrate import solve_ivp

    init_vals = [float(x) for x in init_vals]
    
    if len(init_vals) != number_of_antigens + number_of_antibodies:
        print("""
              Warning, length of init_vals is not equal to total number of
              antigen and antibody variables. Remaining will be set to 0.
              """)

    # create the sustem of equations used to describe the system
    ode_fun, all_variables, inactive_proportions = _systemOfEquations(
        number_of_antigens, number_of_antibodies, association_rates, 
        dissociation_rates, interference_matrix, print_equations=print_equations,
        assay_sensitivity=assay_sensitivity)
    
    number_of_variables = len(all_variables)

    y3 = [0] * (number_of_variables - len(init_vals))

    init_vals = np.array(list(init_vals) + list(y3))

    tspan = [0, measurement_time]
    y = np.zeros((number_of_antigens, len(dilutions)))

    nondiluted_sera_vals = init_vals[number_of_antigens:number_of_antibodies + 1].copy()
    solutions=[]
    for i, dil in enumerate(dilutions):
        init_vals[number_of_antigens:number_of_antibodies + 1] = nondiluted_sera_vals * dil
        t_step = min(measurement_time,600)
        t_eval = np.arange(0, tspan[-1] + t_step,t_step)
       
        sol = solve_ivp(ode_fun, tspan, init_vals, t_eval=t_eval)
        solutions.append(sol.y)
        for j in range(number_of_antigens):
            for k, var in enumerate(all_variables):
                if len(re.findall(r'V' + str(j), var)) > 0: 
                    y[j, i] += (1 - inactive_proportions[k]) * sol.y[k][-1] / init_vals[j]

    log_titers = []
    titers = []
    log2_dilutions = [-np.log2(1 / x) for x in dilutions]
    for j in range(number_of_antigens):
        try:
            a0 = np.argwhere(y[j, :] >= 0.5)[-1][0]
        except Exception:
            titers.append('>' + str(int(1 / dilutions[0])))
            log_titers.append(-np.log2(1 / dilutions[0]))
            continue
        try:
            a1 = np.argwhere(y[j, :] < 0.5)[0][0]
        except Exception:
            titers.append('<' + str(int(1 / dilutions[-1])))
            log_titers.append(-np.log2(1 / dilutions[-1]))
            continue
        val0 = y[j, a0]
        val1 = y[j, a1]

        dil0 = log2_dilutions[a0]
        dil1 = log2_dilutions[a1]

        titer = dil1 + (dil0 - dil1) * (0.5 - val1) / (val0 - val1)
        log_titers.append(titer)
        titers.append(int(2**(-titer)))

    return y, log_titers, titers, solutions


def outputTiterCurvePlot(y, titers, number_of_antigens, dilutions, 
                         markers=None, strain_names=None, fig=None, ax=None):

    """
    Print titer curve for each virus
    """

    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('ggplot')

    if markers is None:
        markers = ['o'] * number_of_antigens

    if strain_names is not None:
        try:
            if len(strain_names) != number_of_antigens:
                raise ValueError("""strain_names should be a list of names
                                  with number of elements equal to
                                  number_of_antigens""")
        except Exception:
            raise ValueError("""strain_names should be a list of names with
                              number_of_antigens elements""")

    if fig is None or ax is None:

        fig, ax = plt.subplots()

    plot_dilutions = [-np.log2(1 / x) for x in dilutions]
    for i in range(number_of_antigens):

        ax.plot(plot_dilutions, y[i, :], marker=markers[i], linewidth=3)
        ax.set_xticks(plot_dilutions)
        ax.set_xticklabels([1 / x for x in dilutions], rotation=90)

        
        
        if titers is not None:
            titer = titers[i]
            ax.scatter(titer, 0, marker='+', color='black')
        ax.set_ylim([-0.1, 1.1])
        ax.set_yticks(np.arange(0,1.25,0.25))

        if strain_names is not None:
            ax.set_title(strain_names[i])

    return ax, fig


def outputScatterPlot(data_x, data_y, xlabel, ylabel, xticks=None,
                      xlabels=None, yticks=None, ylabels=None):

    """
    Scatter plot of two variables which can be used for comparing things like
    association_rates vs titers obtained (see k_vs_titer_notexcess as an example)
    """

    import matplotlib.pyplot as plt

    plt.style.use('ggplot')
    ss = [19.2, 10.2]

    fig, ax = plt.subplots(figsize=(ss[0], ss[1]))

    ax.scatter(data_x, data_y)

    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=90)

    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax, fig
