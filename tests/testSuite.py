#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:03:19 2020

@author: Sina
"""

import sys
import numpy as np
sys.path.append('../')
from NeutralizationBench import _systemOfEquations,titrateAntigensAgainstSera


def test_systemOfEquations(fail_switch=False):
    is_failed=True #every test fails until proven otherwise
    fail_test=0
    if fail_switch:
        fail_test=np.random.randint(4)+1 #fail one of the 4 tests on purpose
    
    
    
    #test1: 0 antigens or antibodies return None
    
    number_of_antigens=0
    number_of_antibodies=0
    if fail_test==1:
        number_of_antigens=1
        number_of_antibodies=1
        
    forward_rates=np.zeros((number_of_antigens,number_of_antibodies))
    backward_ratios=np.zeros((number_of_antigens,number_of_antibodies))
    interference_matrix=np.zeros((number_of_antigens,number_of_antibodies,number_of_antibodies))    
    
    try:
        ode,variables=_systemOfEquations(number_of_antigens,number_of_antibodies,forward_rates,backward_ratios,interference_matrix)   
    except:
        print('Test1 of systemOfEquations failed with exception.')
        return is_failed
        
    if ode!=None or variables!=None:
        if not fail_switch:
            print('Test1 of systemOfEquations failed.')
        return is_failed
    
    

    number_of_antigens=0
    number_of_antibodies=1
        
    forward_rates=np.zeros((number_of_antigens,number_of_antibodies))
    backward_ratios=np.zeros((number_of_antigens,number_of_antibodies))
    interference_matrix=np.zeros((number_of_antigens,number_of_antibodies,number_of_antibodies))
    
    try:
        ode,variables=_systemOfEquations(number_of_antigens,number_of_antibodies,forward_rates,backward_ratios,interference_matrix)   
    except:
       
        print('Test1 of systemOfEquations failed with exception.')
        return is_failed
        
    if ode!=None or variables!=None:
        if not fail_switch:
            print('Test1 of systemOfEquations failed.')
        return is_failed

    

    #test2: 1 antigen, 1 antibody with 0 rate should produce a field with 0 values
    number_of_antigens=1
    number_of_antibodies=1
    
    forward_rates=np.zeros((number_of_antigens,number_of_antibodies))
    backward_ratios=np.zeros((number_of_antigens,number_of_antibodies))
    interference_matrix=np.zeros((number_of_antigens,number_of_antibodies,number_of_antibodies))
    
    if fail_test==2:
        forward_rates=np.ones((number_of_antigens,number_of_antibodies))
   
    ode,variables=_systemOfEquations(number_of_antigens,number_of_antibodies,forward_rates,backward_ratios,interference_matrix)   

    if variables!=['V0','Ab0','V0Ab0'] or not all(x==0 for x in ode(np.random.rand(1,1).flatten(),np.random.rand(1,3).flatten())):
        if not fail_switch:
            print('Test2 of systemOfEquations failed.')
        return is_failed
        
    
    #test3: 1 antigen, 1 antibody with 1 forward rate 1e-4 backward rate 
    number_of_antigens=1
    number_of_antibodies=1
    
    forward_rates=np.ones((number_of_antigens,number_of_antibodies))
    backward_ratios=1e-4*np.ones((number_of_antigens,number_of_antibodies))
    interference_matrix=np.zeros((number_of_antigens,number_of_antibodies,number_of_antibodies))

    if fail_test==3:
        forward_rates=5*np.ones((number_of_antigens,number_of_antibodies))
   
    ode,variables=_systemOfEquations(number_of_antigens,number_of_antibodies,forward_rates,backward_ratios,interference_matrix)   
    z=np.random.rand(1,3).flatten()
    ode_val1=[-z[0]*z[1]+1e-4*z[2],-z[0]*z[1]+1e-4*z[2],z[0]*z[1]-1e-4*z[2]]
    
    ode_val2=ode(np.random.rand(1,1).flatten(),z)
    
    if variables!=['V0','Ab0','V0Ab0'] or not all(x==y for x,y in zip(ode_val1,ode_val2) ):
        if not fail_switch:
            print('Test3 of systemOfEquations failed.')    
        return is_failed
        

    #test4: 1 antigen, 2 antibodies with 1 forward rate 1e-4 backward rate 
    number_of_antigens=1
    number_of_antibodies=2
    if fail_test==4:
        number_of_antibodies=3
        
    forward_rates=np.ones((number_of_antigens,number_of_antibodies))
    backward_ratios=1e-4*np.ones((number_of_antigens,number_of_antibodies))
    interference_matrix=np.ones((number_of_antigens,number_of_antibodies,number_of_antibodies))

    
   
    ode,variables=_systemOfEquations(number_of_antigens,number_of_antibodies,forward_rates,backward_ratios,interference_matrix)   
  
    
    if set(variables)!=set(['V0','Ab0','V0Ab0','V0Ab1','Ab1','Ab1','V0Ab01']):
        if not fail_switch:
            print('Test4 of systemOfEquations failed.')      
        return is_failed
   
    
    return False

def test_titrateAntigensAgainstSera(fail_switch=False):
    
    is_failed=True #every test fails until proven otherwise
    fail_test=0
    if fail_switch:
        fail_test=np.random.randint(2)+1 #fail one of the 2 tests on purpose
    
    
    #test1: 1 antigen and 1 antibody with 0 rates, initial values should remain the same
    number_of_antigens=1
    number_of_antibodies=1
    forward_rates=np.zeros((number_of_antigens,number_of_antibodies))
    backward_ratios=np.zeros((number_of_antigens,number_of_antibodies))
    interference_matrix=np.ones((number_of_antigens,number_of_antibodies,number_of_antibodies))
    
    if fail_test==1:
        forward_rates=np.ones((number_of_antigens,number_of_antibodies))
    

    init_vals=[np.random.randint(100)+10,np.random.randint(100)+10]
    dilutions=[1/2560]
    measurement_time=10
    
    try:
        y,log_titers,titers,sol=titrateAntigensAgainstSera(init_vals,dilutions,number_of_antigens,number_of_antibodies,measurement_time,forward_rates,backward_ratios,interference_matrix,print_equations=False)
    except:
        print('Test1 of titrateAntigensAgainstSera failed with exception.')
        return is_failed
    
    if not all(z==init_vals[0] for z in sol[0,:]) or not all(z==init_vals[1]*dilutions[0] for z in sol[1,:]) or not all(z==sol[2,0] for z in sol[2,:]):
        
        if not fail_switch:
            print('Test1 of titrateAntigensAgainstSera failed.')  
        return is_failed

    
    #test2: 1 antigen and 2 antibodies with second rate 0, second ab should remain constant 
    #others should decrease except complex VAb0 which should increase and VAb1,VAb0Ab1 should all be 0 because Ab1 can not bind
    
    number_of_antigens=1
    number_of_antibodies=2
    forward_rates=np.ones((number_of_antigens,number_of_antibodies))
    
    if fail_test!=2:
        forward_rates[0,1]=0
    
    backward_ratios=np.zeros((number_of_antigens,number_of_antibodies))
    interference_matrix=np.ones((number_of_antigens,number_of_antibodies,number_of_antibodies))
    
    
    init_vals=[np.random.randint(100)+10,np.random.randint(100)+10,np.random.randint(100)+10]
    dilutions=[1/2560]
    measurement_time=10
    
    try:
        y,log_titers,titers,sol=titrateAntigensAgainstSera(init_vals,dilutions,number_of_antigens,number_of_antibodies,measurement_time,forward_rates,backward_ratios,interference_matrix,print_equations=False)
    except:
        print('Test2 of titrateAntigensAgainstSera failed with exception.')
        return is_failed
        
    if not all(z<init_vals[0] for z in sol[0,1:]) or not all(z<init_vals[1]*dilutions[0] for z in sol[1,1:]) or not all(z==init_vals[2]*dilutions[0] for z in sol[2,:]) or not all(z>0 for z in sol[3,1:]) or not all(z==0 for z in sol[4:5,1:].flatten()):
        if not fail_switch:
            print('Test2 of titrateAntigensAgainstSera failed.')              
        return is_failed
    
    
    #test3: 1 antigen and 1 antibody with 0 antibody, initial values should remain the same
    number_of_antigens=1
    number_of_antibodies=1
    forward_rates=np.ones((number_of_antigens,number_of_antibodies))
    backward_ratios=1e-4*np.ones((number_of_antigens,number_of_antibodies))
    interference_matrix=np.ones((number_of_antigens,number_of_antibodies,number_of_antibodies))
    
    if fail_test==3:
        init_vals=[np.random.randint(100)+10,np.random.randint(100)+10]
    else:
        init_vals=[np.random.randint(100)+10,0]
    
    dilutions=[1/2560]
    measurement_time=10
    
    try:
        y,log_titers,titers,sol=titrateAntigensAgainstSera(init_vals,dilutions,number_of_antigens,number_of_antibodies,measurement_time,forward_rates,backward_ratios,interference_matrix,print_equations=False)
    except:
        print('Test3 of titrateAntigensAgainstSera failed with exception.')
        return is_failed
    
    if not all(z==sol[0,0] for z in sol[0,:]) or not all(z==0 for z in sol[1,:]) or not all(z==0 for z in sol[2,:]):
        
        if not fail_switch:
            print('Test3 of titrateAntigensAgainstSera failed.')  
        return is_failed

   
    #test4: 1 antigen and 2 antibodies with second antibody 0 concentration, second ab should remain constant 
    #others should decrease except complex VAb0 which should increase and VAb1,VAb0Ab1 should all be 0 because Ab1 can not bind
    
    number_of_antigens=1
    number_of_antibodies=2
    forward_rates=np.ones((number_of_antigens,number_of_antibodies))
    backward_ratios=np.zeros((number_of_antigens,number_of_antibodies))
    interference_matrix=np.ones((number_of_antigens,number_of_antibodies,number_of_antibodies))
    
    
    if fail_test==4:
        init_vals=[np.random.randint(100)+10,np.random.randint(100)+10,np.random.randint(100)+10]
    else:
        init_vals=[np.random.randint(100)+10,np.random.randint(100)+10,0]
    
    dilutions=[1/2560]
    measurement_time=10
    
    try:
        y,log_titers,titers,sol=titrateAntigensAgainstSera(init_vals,dilutions,number_of_antigens,number_of_antibodies,measurement_time,forward_rates,backward_ratios,interference_matrix,print_equations=False)
    except:
        print('Test4 of titrateAntigensAgainstSera failed with exception.')
        return is_failed
        
    if not all(z<init_vals[0] for z in sol[0,1:]) or not all(z<init_vals[1]*dilutions[0] for z in sol[1,1:]) or not all(z==init_vals[2]*dilutions[0] for z in sol[2,:]) or not all(z>0 for z in sol[3,1:]) or not all(z==0 for z in sol[4:5,1:].flatten()):
        if not fail_switch:
            print('Test4 of titrateAntigensAgainstSera failed.')              
        return is_failed
    
    
    return False
    

