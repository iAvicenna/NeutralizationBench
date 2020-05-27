#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:24:48 2020

@author: Sina
"""

from testSuite import test_systemOfEquations, test_titrateAntigensAgainstSera, test_ConservationLaws, test_interferenceMatrix, test_inactivationThresholds
import time

# There is some randomness in tests so we iterate each test multiple times
# to make sure each test is succesful for random inputs
# When fail_switch is on, the inputs to the tests are messed with in such a way
# that tests should fail.

failed_test = 0
print('Testing function inactivation_thresholds parameter (2 tests)')
for i in range(5):    
    is_failed = test_inactivationThresholds()
    if is_failed:
        print('test_inactivationThresholds failed.')
        failed_test += 1
        break
    
    is_failed =  test_inactivationThresholds(fail_switch=True)
    if not is_failed:
        print('test_inactivationThresholds with fail_switch On did not fail.')
        failed_test += 1
        break
        

t0 = time.time()
print('Testing function systemOfEquations (4 tests)')
for i in range(5):
    is_failed = test_systemOfEquations()
    if is_failed:
        print('test_systemOfEquations failed.')
        failed_test += 1
        break

    is_failed = test_systemOfEquations(fail_switch=True)
    if not is_failed:
        print('test_systemOfEquations with fail_switch On did not fail.')
        failed_test += 1
        break

print('Testing Conservation Laws for Vector Fields and Solutions (2 tests)')
for i in range(5):    
    is_failed = test_ConservationLaws()
    if is_failed:
        print('test_ConservationLaws failed.')
        failed_test += 1
        break
    
    is_failed = test_ConservationLaws(fail_switch=True)
    if not is_failed:
        print('test_ConservationLaws with fail_switch On did not fail.')
        failed_test += 1
        break
        
        
print('Testing function titrateAntigensAgainstSera (5 tests)')    
for i in range(5):    
    is_failed = test_titrateAntigensAgainstSera()
    if is_failed:
        print('test_titrateAntigensAgainstSera failed.')
        failed_test += 1
        break
    is_failed = test_titrateAntigensAgainstSera(fail_switch=True)
    if not is_failed:
        print('test_titrateAntigensAgainstSera with fail_switch On did not fail.')
        failed_test += 1
        break


print('Testing interference_matrix Parameter (2 tests)')    
for i in range(10):    
    is_failed = test_interferenceMatrix()
    if is_failed:
        print('test_interferenceMatrix failed.')
        failed_test += 1
        break
    
    is_failed = test_interferenceMatrix(fail_switch=True)
    if not is_failed:
        print('test_interferenceMatrix with fail_switch On did not fail.')
        failed_test += 1
        break

t1 = time.time()

if failed_test == 0:
    print('All tests succesful ({:.2f} seconds)'.format(t1 - t0))
else:
    print('WARNING: Some tests failed ({:.2f} seconds)'.format(t1 - t0))
