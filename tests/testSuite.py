#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:03:19 2020

@author: Sina
"""

import sys
import numpy as np
sys.path.append('../')
from NeutralizationBench import _systemOfEquations, titrateAntigensAgainstSera


def test_systemOfEquations(fail_switch=False):
    is_failed = True  # every test fails until proven otherwise
    fail_test = 0
    if fail_switch:
        fail_test = np.random.randint(4) + 1  # fail one of the 4 tests on purpose

    #  test1: 0 antigens or antibodies return None
    
    number_of_antigens = 0
    number_of_antibodies = 0
    if fail_test == 1:
        number_of_antigens = 1
        number_of_antibodies = 1
        
    association_rates = np.zeros((number_of_antigens, number_of_antibodies))
    dissociation_constants = np.zeros((number_of_antigens, number_of_antibodies))
    interference_matrix = np.zeros((
        number_of_antigens, number_of_antibodies, number_of_antibodies))    
    
    try:
        ode, variables = _systemOfEquations(
            number_of_antigens, number_of_antibodies, association_rates,
            dissociation_constants, interference_matrix)   
    except Exception as e:
        print('Test1 of systemOfEquations failed with exception\n "{}".'.format(e))
        return is_failed
        
    if ode is not None or variables is not None:
        if not fail_switch:
            print('Test1 of systemOfEquations failed.')
        return is_failed
    
    number_of_antigens = 0
    number_of_antibodies = 1
        
    association_rates = np.zeros((number_of_antigens, number_of_antibodies))
    dissociation_constants = np.zeros((number_of_antigens, number_of_antibodies))
    interference_matrix = np.zeros((number_of_antigens, number_of_antibodies,
                                    number_of_antibodies))
    
    try:
        ode, variables = _systemOfEquations(
            number_of_antigens, number_of_antibodies, association_rates,
            dissociation_constants, interference_matrix)   
    except Exception as e:
       
        print('Test1 of systemOfEquations failed with exception\n {} .'.format(e))
        return is_failed
        
    if ode is not None or variables is not None:
        if not fail_switch:
            print('Test1 of systemOfEquations failed.')
        return is_failed

    # test2: 1 antigen, 1 antibody with 0 rate should produce a field with 0 values
    number_of_antigens = 1
    number_of_antibodies = 1
    
    association_rates = np.zeros((number_of_antigens, number_of_antibodies))
    dissociation_constants = np.zeros((number_of_antigens, number_of_antibodies))
    interference_matrix = np.zeros((number_of_antigens, number_of_antibodies,
                                    number_of_antibodies))
    
    if fail_test == 2:
        association_rates = np.ones((number_of_antigens, number_of_antibodies))
   
    try:
        ode, variables = _systemOfEquations(
            number_of_antigens, number_of_antibodies, association_rates,
            dissociation_constants, interference_matrix)   
        
    except Exception as e:
       
        print('Test2 of systemOfEquations failed with exception\n"{}".'.format(e))
        return is_failed
        
    if variables != ['V0', 'Ab_0', 'V0Ab_0'] or not all(
            x == 0 for x in ode(np.random.rand(1, 1).flatten(),
                                np.random.rand(1, 3).flatten())):
        
        if not fail_switch:
            print('Test2 of systemOfEquations failed.')
        return is_failed
        
    # test3: 1 antigen, 1 antibody with 1 forward rate 1e-4 backward rate 
    number_of_antigens = 1
    number_of_antibodies = 1
    
    association_rates = np.ones((number_of_antigens, number_of_antibodies))
    dissociation_constants = 1e-4 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies,
                                   number_of_antibodies))

    if fail_test == 3:
        association_rates = 5 * np.ones((number_of_antigens, number_of_antibodies))
   
    try:
        ode, variables = _systemOfEquations(
            number_of_antigens, number_of_antibodies, association_rates,
            dissociation_constants, interference_matrix)   
        
    except Exception as e:
       
        print('Test3 of systemOfEquations failed with exception\n"{}".'.format(e))
        return is_failed
        
    z = np.random.rand(1, 3).flatten()
    ode_val1 = [-z[0] * z[1] + 1e-4 * z[2], -z[0] * z[1] + 1e-4 * z[2],
                z[0] * z[1] - 1e-4 * z[2]]
    
    ode_val2 = ode(np.random.rand(1, 1).flatten(), z)
    
    if variables != ['V0', 'Ab_0', 'V0Ab_0'] or not all(x == y for x, y in 
                                                        zip(ode_val1, ode_val2)):
        if not fail_switch:
            print('Test3 of systemOfEquations failed.')    
        return is_failed
        
    # test4: 1 antigen, 2 antibodies with 1 forward rate 1e-4 backward rate 
    number_of_antigens = 1
    number_of_antibodies = 2
    if fail_test == 4:
        number_of_antibodies = 3
        
    association_rates = np.ones((number_of_antigens, number_of_antibodies))
    dissociation_constants = 1e-4 * np.ones((number_of_antigens,
                                      number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, 
                                  number_of_antibodies,
                                  number_of_antibodies))

    try:
        ode, variables = _systemOfEquations(
            number_of_antigens, number_of_antibodies, association_rates,
            dissociation_constants, interference_matrix)   
        
    except Exception as e:
       
        print('Test4 of systemOfEquations failed with exception\n"{}".'.format(e))
        return is_failed
         
    if set(variables) != set(['V0', 'Ab_0', 'V0Ab_0', 'V0Ab_1', 'Ab_1',
                              'Ab_1', 'V0Ab_0_1']):
        if not fail_switch:
            print('Test4 of systemOfEquations failed.')      
        return is_failed
   
    if fail_switch:
        print('Test{} of systemOfEquations did not fail with fail_switch on.'.format(fail_test))
    
    return False


def test_ConservationLaws(fail_switch=False):
    
    is_failed = True  # every test fails until proven otherwise
    fail_test = 0
    if fail_switch:
        fail_test = np.random.randint(3) + 1  # fail one of the 3 tests on purpose
    
    # test1: 2 antigen, 2 antibody,d(V-Ab)/dt = d(V+AbV)/dt = dAb+AbV/dt=0
    number_of_antigens = 1
    number_of_antibodies = 1
    rand1 = np.random.rand()
    rand2 = np.random.rand()
    
    association_rates = np.ones((number_of_antigens, number_of_antibodies))
    dissociation_constants = np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies,
                                   number_of_antibodies))

    try:
        ode, variables = _systemOfEquations(
            number_of_antigens, number_of_antibodies, association_rates,
            dissociation_constants, interference_matrix)   
        
    except Exception as e:
       
        print('Test1 of conservationLaws failed with exception\n"{}".'.format(e))
        return is_failed    
          
    z = np.random.rand(1, 3).flatten()
    t = np.random.rand(1, 1).flatten()
    
    ode_val = ode(t, z)  
    
    if fail_test == 1:
        ode_val[0] += 10
    
    if (np.abs(ode_val[0] - ode_val[1]) > 1e-5 or np.abs(ode_val[0] + ode_val[2]) > 1e-5 
            or np.abs(ode_val[1] + ode_val[2]) > 1e-5):
        
        if not fail_switch:
            print('Test1 of ConservationLaws failed.')
        
        return is_failed
        
    # test2: 2 antigen, 2 antibody,d(V-Ab)/dt = d(V+AbV)/dt = dAb+AbV/dt=0
    number_of_antigens = 2
    number_of_antibodies = 2
    rand1 = np.random.rand() + 0.1
    rand2 = np.random.rand() + 0.1
    
    association_rates = rand1 * np.ones((number_of_antigens, number_of_antibodies))
    dissociation_constants = rand2 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies,
                                   number_of_antibodies))

    try:
        ode, variables = _systemOfEquations(
            number_of_antigens, number_of_antibodies, association_rates,
            dissociation_constants, interference_matrix)   
        
    except Exception as e:
       
        print('Test2 of conservationLaws failed with exception\n"{}".'.format(e))
        return is_failed       
        
    z = np.random.rand(1, 10).flatten()
    t = np.random.rand(1, 1).flatten()
    
    ode_val = ode(t, z)  
    
    if fail_test == 2:
        ode_val[0] += 10
    
    sum1 = ode_val[0] + ode_val[1] - ode_val[2] - ode_val[3] - ode_val[6] - ode_val[9]
    
    if sum1 > 1e-5:
          
        if not fail_switch:
            print('Test2 of ConservationLaws failed.')
        return is_failed
    
    # test3: solutions of the equations should also satisfy the same conservation laws:
    init_vals = [np.random.randint(100) + 10, np.random.randint(100) + 10, 
                 np.random.randint(100) + 10, np.random.randint(100) + 10]
    dilutions = [1 / 2560]
    measurement_time = 100
    
    try:
        y, log_titers, titers, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants, interference_matrix,
            print_equations=False)
        
    except Exception as e:
       
        print('Test3 of conservationLaws failed with exception\n"{}".'.format(e))
        return is_failed   
    
    if fail_test == 3:
        sol[0][0, 0] *= 10
    
    sum1 = sol[0][0, 0] + sol[0][1, 0] - sol[0][2, 0] - sol[0][3, 0] - sol[0][6, 0] - sol[0][9, 0]
    
    sum2 = sol[0][0, -1] + sol[0][1, -1] - sol[0][2, -1] - sol[0][3, -1] - sol[0][6, -1] - sol[0][9, -1]
    
    if np.abs(sum1 - sum2) > 1e-5:
          
        if not fail_switch:
            print('Test3 of ConservationLaws failed.')
        return is_failed
    
    if fail_switch:
        print('Test{} of Conservationlaws did not fail with fail_switch on.'.format(fail_test))
    
    return False
 
    

def test_titrateAntigensAgainstSera(fail_switch=False):
    
    M = 6e23
    vratio = 50
    nspike = 450
    ratio = vratio * nspike
    total_volume = 1e-4
    total_antibody = (np.random.rand(1,1)+0.5) * 600 * 5e9
    total_PFU=(np.random.rand(1,1)+0.5) * 1000


    
    is_failed = True  # every test fails until proven otherwise
    fail_test = 0
    if fail_switch:
        fail_test = np.random.randint(5) + 1  # fail one of the 5 tests on purpose
        
    # test1: 1 antigen and 1 antibody with 0 rates, initial values should remain the same
    number_of_antigens = 1
    number_of_antibodies = 1
    association_rates = np.zeros((number_of_antigens, number_of_antibodies)) * 10e6 / M
    dissociation_constants = np.zeros((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((
        number_of_antigens, number_of_antibodies, number_of_antibodies))
    
    if fail_test == 1:
        association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M
    
    init_vals = [total_PFU * ratio / total_volume, total_antibody / total_volume]
    dilutions = [1 / 2560]
    measurement_time = 10
    
    try:
        y, log_titers, titers, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants, 
            interference_matrix, print_equations=False)
    except Exception as e:
        print('Test1 of titrateAntigensAgainstSera failed with exception\n "{}".'.format(e))
        return is_failed
    
    if (not all(z == init_vals[0] for z in sol[0][0, :]) or 
            not all(z == init_vals[1] * dilutions[0] for z in sol[0][1, :]) 
            or not all(z == sol[0][2, 0] for z in sol[0][2, :])):
            
        if not fail_switch:
            print('Test1 of titrateAntigensAgainstSera failed.')  
        return is_failed

    # test2: 1 antigen and 2 antibodies with second rate 0, second ab should remain constant 
    # others should decrease except complex VAb0 which should increase and VAb1,VAb0Ab1 should all be 0 because Ab1 can not bind
    
    number_of_antigens = 1
    number_of_antibodies = 2
    association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M
    
    if fail_test != 2:
        association_rates[0, 1] = 0
    
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies,
                                   number_of_antibodies))
    
    init_vals = [total_PFU * ratio / total_volume, total_antibody / total_volume, total_antibody / total_volume]

    
    dilutions = [1 / 2560]
    measurement_time = 10
    
    try:
        y, log_titers, titers, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants,
            interference_matrix, print_equations=False)
    except Exception as e:
        print('Test2 of titrateAntigensAgainstSera failed with exception\n "{}".'.format(e))
        return is_failed
        
    if (not all(z < init_vals[0] for z in sol[0][0, 1:]) or 
            not all(z < init_vals[1] * dilutions[0] for z in sol[0][1, 1:]) or not 
            all(np.abs(z - init_vals[2] * dilutions[0]) < 1e-5 for z in sol[0][2, :]) or not 
            all(z > 0 for z in sol[0][3, 1:]) or not 
            all(z == 0 for z in sol[0][4:5, 1:].flatten())):
            
        if not fail_switch:
            print('Test2 of titrateAntigensAgainstSera failed.')              
        return is_failed
    
    # test3: 1 antigen and 1 antibody with 0 antibody, initial values should remain the same
    
    number_of_antigens = 1
    number_of_antibodies = 1
    association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
    
    if fail_test == 3:
        init_vals = [total_PFU * ratio / total_volume, total_antibody / total_volume]
    else:
        init_vals = [total_PFU * ratio / total_volume, 0]
    
    dilutions = [1 / 2560]
    measurement_time = 10
    
    try:
        y, log_titers, titers, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants, 
            interference_matrix, print_equations=False)
    except Exception as e:
        print('Test3 of titrateAntigensAgainstSera failed with exception\n "{}".'.format(e))
        return is_failed
    
    if (not all(np.abs(z - sol[0][0, 0]) < 1e-5 for z in sol[0][0, :]) or not 
            all(np.abs(z) < 1e-5 for z in sol[0][1, :]) or not all(np.abs(z) < 1e-5 for z in sol[0][2, :])):
        
        if not fail_switch:
            print('Test3 of titrateAntigensAgainstSera failed.')  
        return is_failed

    # test4: 1 antigen and 2 antibodies with second antibody 0 concentration, second ab should remain constant 
    # others should decrease except complex VAb0 which should increase and VAb1,VAb0Ab1 should all be 0 because Ab1 can not bind
    
    number_of_antigens = 1
    number_of_antibodies = 2
    association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies,
                                   number_of_antibodies))
    
    if fail_test == 4:
        init_vals = [total_PFU * ratio / total_volume,
                     total_antibody / total_volume, total_antibody / total_volume]
    else:
        init_vals = [total_PFU * ratio / total_volume,
                     total_antibody / total_volume, 0]
    
    dilutions = [1 / 2560]
    measurement_time = 10
    
    try:
        y, log_titers, titers, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants, 
            interference_matrix, print_equations=False)
    except Exception as e:
        print('Test4 of titrateAntigensAgainstSera failed with exception\n "{}".'.format(e))
        return is_failed
        
    if (not all(z < init_vals[0] for z in sol[0][0, 1:]) or not 
            all(z < init_vals[1] * dilutions[0] for z in sol[0][1, 1:]) or not 
            all(z == init_vals[2] * dilutions[0] for z in sol[0][2, :]) or not 
            all(z > 0 for z in sol[0][3, 1:]) or not 
            all(z == 0 for z in sol[0][4:5, 1:].flatten())):
        
        if not fail_switch:
            print('Test4 of titrateAntigensAgainstSera failed.')              
        return is_failed
    
    #test5 mixing 2 antibodies with same k should have the same effect as
    #one antibody with double the concentration
    
    number_of_antigens = 1
    number_of_antibodies = 2
    association_rates = 0.5 * np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M 
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
    
    if fail_test == 5:
        init_vals = [total_PFU * ratio / total_volume, total_antibody / total_volume, 0.1 * total_antibody / total_volume]
    else:
        init_vals = [total_PFU * ratio / total_volume, total_antibody / total_volume, total_antibody / total_volume]
    
    dilutions = [1 / 5120, 1 / 1280]
    measurement_time = 60
    
    try:
        y1, log_titers, titers1, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants, 
            interference_matrix, print_equations=False)
    except Exception as e:
        print('Test5 of titrateAntigensAgainstSera failed with exception\n "{}".'.format(e))
        return is_failed
    
    
    number_of_antigens = 1
    number_of_antibodies = 1
    association_rates = 0.5*np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
    
    init_vals = [total_PFU * ratio / total_volume, 2*total_antibody / total_volume]
    
    try:
        y2, log_titers, titers2, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants, 
            interference_matrix, print_equations=False)
    except Exception as e:
        print('Test5 of titrateAntigensAgainstSera failed with exception\n "{}".'.format(e))
        return is_failed
    
    
    if not any(np.abs(x - y) < 1e-3 for x, y in zip(y1.flatten(), y2.flatten())):
        if not fail_switch:
            print('Test5 of titrateAntigensAgainstSera failed.')              
        return is_failed
    
    
    if fail_switch:
        print('Test{} of titrateAntigensAgainstSera did not fail with fail_switch on.'.format(fail_test))
    
    return False

def test_inactivationThresholds(fail_switch=False):
    M = 6e23
    vratio = 50
    nspike = 450
    ratio = vratio * nspike
    total_volume = 1e-4
    total_antibody = (np.random.rand()+0.5) * 600 * 5e9
    total_PFU=(np.random.rand()+0.5) * 1000

    
    is_failed = True  # every test fails until proven otherwise
    fail_test = 0
    if fail_switch:
        fail_test = np.random.randint(2) + 1  # fail one of the 2 tests on purpose
    
    # test 1: two antigens and one antibody which can only neutralize if number
    # of bound antibodies per site is = max which is practically impossible 
    # so should have no titers
        
    number_of_antigens = 2
    number_of_antibodies = 1
    a1 = np.random.randint(-4,4) 
    dilutions = 1 / np.array([1280,640,320,160,80,40,20])
    measurement_time = 3600
    association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 2 ** a1 * 1e5 / M
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
    init_vals = [total_PFU * ratio / total_volume, total_PFU * ratio / total_volume,
                 2*total_antibody / total_volume]

    if fail_test == 1:
        inactivation_thresholds=[1]
    else:
        inactivation_thresholds=[150]
        
    try:
        y, log_titers, titers, sol = titrateAntigensAgainstSera(
        init_vals, dilutions, number_of_antigens, number_of_antibodies,
        measurement_time, association_rates, dissociation_constants, 
        interference_matrix, inactivation_thresholds = inactivation_thresholds,
        print_equations=False)
    except Exception as e:
        print('Test1 of inactivationThresholds failed with exception\n "{}".'.format(e))
        return is_failed    
       
    if not all(np.abs(x-1)<1e-6 for x in y.flatten()) and not all(x == '<20' for x in titers):
        if not fail_switch:
            print('Test1 of inactivationThresholds failed.')  
        return is_failed
    
    
    number_of_antigens = 1
    number_of_antibodies = 1
    a1 = 0
    dilutions = 1 / np.array([2560, 1280, 640, 320, 160, 80, 40, 20])
    measurement_time = 3600
    association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 2 ** a1 * 1e5 / M
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
    init_vals = [total_PFU * ratio / total_volume,  2*total_antibody / total_volume]

    if fail_test == 2:
        inactivation_thresholds=[1]
    else:
        inactivation_thresholds=[2]
        
    try:
        y, log_titers, titers1, sol = titrateAntigensAgainstSera(
        init_vals, dilutions, number_of_antigens, number_of_antibodies,
        measurement_time, association_rates, dissociation_constants, 
        interference_matrix, inactivation_thresholds = inactivation_thresholds,
        print_equations=False)
        
        inactivation_thresholds=[1]
        
        y, log_titers, titers2, sol = titrateAntigensAgainstSera(
        init_vals, dilutions, number_of_antigens, number_of_antibodies,
        measurement_time, association_rates, dissociation_constants, 
        interference_matrix, inactivation_thresholds = inactivation_thresholds,
        print_equations=False)
    except Exception as e:
        print('Test2 of inactivationThresholds failed with exception\n "{}".'.format(e))
        return is_failed    
    
    
    if isinstance(titers2[0],str):
        if '<' in titers2[0]:
            titers2[0] = 0.5*int(titers2[0][1:])
        elif '>' in titers2[0]:
            titers2[0] = 2*int(titers2[0][1:])  
    
    if isinstance(titers1[0],str):        
        if '<' in titers1[0]:
            titers1[0] = 0.5*int(titers1[0][1:])
        elif '>' in titers1[0]:
            titers1[0] = 2*int(titers1[0][1:])      
        
    if titers2[0] <= titers1[0]:
        if not fail_switch:
            print('Test2 of inactivationThresholds failed.')  
        return is_failed
   
    return False
   

def test_interferenceMatrix(fail_switch=False):
    
    M = 6e23
    vratio = 50
    nspike = 450
    ratio = vratio * nspike
    total_volume = 1e-4
    total_antibody = (np.random.rand(1,1)+0.5) * 600 * 5e9
    total_PFU=(np.random.rand(1,1)+0.5) * 1000
    
    is_failed = True  # every test fails until proven otherwise
    fail_test = 0
    if fail_switch:
        fail_test = np.random.randint(3) + 1  # fail one of the 2 tests on purpose
    
    # test1: 1 antigen and 2 antibody with complete interference so there should
    # be no V_0Ab_0_1
    
    number_of_antigens = 1
    number_of_antibodies = 2
    association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    
    init_vals = [total_PFU * ratio / total_volume,
                 2*total_antibody / total_volume, 2*total_antibody / total_volume]

    if fail_test == 1:
        interference_matrix = np.ones((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
    else:
        interference_matrix = np.zeros((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
    
    dilutions = 1 / np.array([5120])
    measurement_time = 50
    
    try:
        y, log_titers, titers, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants, 
            interference_matrix, print_equations=False)
    except Exception as e:
        print('Test1 of interference_matrix failed with exception\n "{}".'.format(e))
        return is_failed
    
    if not all(np.abs(z) == 0 for z in sol[0][5,:].flatten()):
        
        if not fail_switch:
            print('Test1 of titrateAntigensAgainstSera failed.')  
        return is_failed
    
    # test2: 1 antigen and 3 antibody with complete interference between 1 and 2
    # so there should be no V_0Ab_0_1, V_0Ab_0_1_2 but there should be others
    
    number_of_antigens = 1
    number_of_antibodies = 3
    association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    
    init_vals = [total_PFU * ratio / total_volume, 2*total_antibody / total_volume, 
                 2*total_antibody / total_volume, 2*total_antibody / total_volume]

    if fail_test == 2:
        interference_matrix = np.ones((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
    else:
        interference_matrix = np.ones((number_of_antigens, number_of_antibodies, 
                                   number_of_antibodies))
        interference_matrix[0, 0, 1] = 0
        interference_matrix[0, 1, 0] = 0
    
    dilutions = 1 / np.array([320])
    measurement_time = 50
    
    try:
        y, log_titers, titers, sol = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_rates, dissociation_constants, 
            interference_matrix, print_equations=False)
    except Exception as e:
        print('Test2 of interference_matrix failed with exception\n "{}".'.format(e))
        return is_failed
    
    if (not all(np.abs(z) == 0 for z in sol[0][7,:].flatten()) 
        or not all(np.abs(z) == 0 for z in sol[0][10,:].flatten())
        or not all(np.abs(z) != 0 for z in sol[0][8:9,-1].flatten())
        or not all(np.abs(z) != 0 for z in sol[0][5:6,-1].flatten())):
        
        if not fail_switch:
            print('Test2 of interference_matrix failed.')  
        return is_failed
    
   
    
    # test3: an antibody with high association constant but low neutralization
    # should decrease titers
    
    number_of_antigens = 1
    number_of_antibodies = 2    
    association_constants = np.ones((number_of_antigens, number_of_antibodies)) * 0.1e5 / M
    association_constants[0,1]*=100
    total_antibody = 5e13
    
    dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies,
                                   number_of_antibodies))
    
    interference_matrix[0, 0, 1] = 0 #complete interference
    interference_matrix[0, 1, 0] = 0
    inactivation_thresholds=[1, 150]
    total_PFU = 1000
    init_vals = [total_PFU * ratio / total_volume, 0.5 * total_antibody / total_volume, 0.5 * total_antibody / total_volume]
    dilutions = 1 / np.array([5120, 2560, 1280, 640, 320, 160, 80, 40, 20])
    measurement_time = 3600
    
    interference_matrix[0, 0, 1] = 0 #complete interference
    interference_matrix[0, 1, 0] = 0
    
    if fail_test == 3:
        interference_matrix[0, 0, 1] = 1 
        interference_matrix[0, 1, 0] = 1
    
    try:
            
        y1, log_titers1, titers, sol1 = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_constants, dissociation_constants, interference_matrix,
            inactivation_thresholds=inactivation_thresholds)   
        
        interference_matrix[0, 0, 1] = 1 #no interference
        interference_matrix[0, 1, 0] = 1
        y2, log_titers2, titers2,sol2 = titrateAntigensAgainstSera(
            init_vals, dilutions, number_of_antigens, number_of_antibodies,
            measurement_time, association_constants, dissociation_constants, interference_matrix,
            inactivation_thresholds=inactivation_thresholds)  
    except Exception as e:
        print('Test3 of interference_matrix failed with exception\n "{}".'.format(e))
        return is_failed
    
    if any(x<=y for x,y in zip(y1.flatten(),y2.flatten())):
        if not fail_switch:
            print('Test3 of interference_matrix failed.')  
        return is_failed
    
    if fail_switch:
        print('Test{} of interference_matrix did not fail with fail_switch on.'.format(fail_test))
    
    return False
    
