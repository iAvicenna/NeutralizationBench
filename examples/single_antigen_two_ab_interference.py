#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:41:57 2020

@author: Sina

testing one antigens single sera when there is complete interference
"""
import sys
import numpy as np    
sys.path.append("../")
from NeutralizationBench import titrateAntigensAgainstSera
from NeutralizationBench import outputTiterCurvePlot


#  Ab2 is a strong binder but not a very good inactivator. When it interferes
#  with the binding of others, it decreases titer.

number_of_antigens = 1
number_of_antibodies = 2
M = 6e23
particle_to_pfu_ratio = 50
nspikes = 150
total_volume = 1e-4

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
init_vals = [total_PFU * particle_to_pfu_ratio * nspikes / total_volume, 0.5 * total_antibody / total_volume, 0.5 * total_antibody / total_volume]
dilutions = 1 / np.array([5120, 2560, 1280, 640, 320, 160, 80, 40, 20])
measurement_time = 3600

interference_matrix[0, 0, 1] = 0 #complete interference
interference_matrix[0, 1, 0] = 0

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

ax = None
fig = None

ax, fig = outputTiterCurvePlot(
    y1, log_titers1, number_of_antigens, dilutions, fig=fig, ax=ax)


ax, fig = outputTiterCurvePlot(
    y2, log_titers2, number_of_antigens, dilutions, fig=fig, ax=ax)
