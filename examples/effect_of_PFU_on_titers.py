#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:41:57 2020

@author: Sina
"""
import numpy as np    
import sys
sys.path.append("../")
from NeutralizationBench import titrateAntigensAgainstSera
from NeutralizationBench import outputTiterCurvePlot


number_of_antigens = 1
number_of_antibodies = 1
M = 6e23
vratio = 50
nspike = 450
ratio = vratio * nspike
total_volume = 1e-4


# with these parameters, pfu change leads to titer shift
# forward_rates=np.ones((number_of_antigens,number_of_sera))*100e6/(M)
# total_antibody=4*5e9

# with these parameters, pfu change does not lead to titer shift
# total_antibody=600*5e9
# forward_rates=np.ones((number_of_antigens,number_of_sera))*10e6/(M)


forward_rates = np.ones((number_of_antigens, number_of_antibodies)) * 10e6 / M
total_antibody = 600 * 5e9

backward_ratios = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
interference_matrix = np.ones((number_of_antigens, number_of_antibodies, number_of_antibodies))

total_PFUs = [100, 10000]

ax = None
fig = None

for total_PFU in total_PFUs:  
    init_vals = [total_PFU * ratio / total_volume, total_antibody / total_volume]
    
    measurement_time = 3600
    dilutions = 1 / np.array([5120, 2560, 1280, 640, 320, 160, 80, 40, 20])
    
    y, log_titer, titer, _ = titrateAntigensAgainstSera(
        init_vals, dilutions, 
        number_of_antigens, number_of_antibodies, 
        measurement_time, forward_rates, backward_ratios, 
        interference_matrix)   
    
    ax, fig = outputTiterCurvePlot(y, log_titer, number_of_antigens, dilutions,
                                   fig=fig, ax=ax)

    ax.set_ylabel('Amount relative to initial PFU=100')
    ax.set_xlabel('Dilution')
    
ax.legend(['100 PFU', '1000 PFU'])
ax.set_title('High reactivity strain')
