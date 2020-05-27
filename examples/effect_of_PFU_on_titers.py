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
particle_to_pfu_ratio = 50
nspikes = 150
total_volume = 1e-4


# with these parameters, pfu change leads to titer shift
# association_rates=np.ones((number_of_antigens,number_of_sera))*5e8/(M)
# total_antibody=5e8


# with these parameters, pfu change does not lead to titer shift
# total_antibody=600*5e9
# association_rates=np.ones((number_of_antigens,number_of_sera))*1e5/(M)


association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 1e5 / M
total_antibody = 600*5e9

dissociation_rates = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
interference_matrix = np.ones((number_of_antigens, number_of_antibodies, number_of_antibodies))

total_PFUs = [100, 10000]

ax = None
fig = None

for total_PFU in total_PFUs:  
    init_vals = [total_PFU * particle_to_pfu_ratio * nspikes, total_antibody / total_volume]
    
    measurement_time = 3600
    dilutions = 1 / np.array([5120, 2560, 1280, 640, 320, 160, 80, 40, 20])
    
    y, log_titer, titer, _ = titrateAntigensAgainstSera(
        init_vals, dilutions, 
        number_of_antigens, number_of_antibodies, 
        measurement_time, association_rates, dissociation_rates, 
        interference_matrix)   
    
    ax, fig = outputTiterCurvePlot(y, log_titer, number_of_antigens, dilutions,
                                   fig=fig, ax=ax)

    ax.set_ylabel('Amount relative to initial PFU=100')
    ax.set_xlabel('Dilution')
    
ax.legend(['100 PFU', '1000 PFU'])
ax.set_title('High reactivity strain')
