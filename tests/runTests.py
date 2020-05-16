#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:24:48 2020

@author: Sina
"""

from testSuite import test_systemOfEquations,test_titrateAntigensAgainstSera
import time

#There is some randomness in tests so we iterate each test multiple times
#to make sure each test is succesful for random inputs
#When fail_switch is on, the inputs to the tests are messed with in such a way
#that tests should fail.


failed_test=0
t0=time.time()
for i in range(50):
    is_failed=test_systemOfEquations()
    if is_failed==True:
        print('test_systemOfEquations failed.')
        failed_test+=1

    is_failed=test_systemOfEquations(fail_switch=True)
    if is_failed==False:
        print('test_systemOfEquations with fail_switch On did not fail.')
        failed_test+=1
    
for i in range(10):    
    is_failed=test_titrateAntigensAgainstSera()
    if is_failed==True:
        print('test_titrateAntigensAgainstSera failed.')
        failed_test+=1
    
    is_failed=test_titrateAntigensAgainstSera(fail_switch=True)
    if is_failed==False:
        print('test_titrateAntigensAgainstSera with fail_switch On did not fail.')
        failed_test+=1


t1=time.time()

if failed_test==0:
    print('All tests succesful ({:.2f} seconds)'.format(t1-t0))
else:
    print('WARNING: Some tests failed ({:.2f} seconds)'.format(t1-t0))    