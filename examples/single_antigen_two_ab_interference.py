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


number_of_antigens = 1
number_of_antibodies = 2
M = 6e23
vratio = 50
nspike = 450
ratio = vratio * nspike
total_volume = 1e-4

association_constants = np.ones((number_of_antigens, number_of_antibodies)) * 5e5 / M
total_antibody = 5e13
association_constants[0, 1] *= 5

dissociation_constants = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
interference_matrix = np.ones((number_of_antigens, number_of_antibodies,
                               number_of_antibodies))

interference_matrix[0, 0, 1] = 0 #complete interference
interference_matrix[0, 1, 0] = 0

total_PFU = 1000
init_vals = [total_PFU * ratio / total_volume, 0.5 * total_antibody / total_volume, 0.5 * total_antibody / total_volume]
dilutions = 1 / np.array([5120, 2560, 1280, 640, 320, 160, 80, 40, 20])
measurement_time = 36


y, log_titers, titers, _ = titrateAntigensAgainstSera(
    init_vals, dilutions, number_of_antigens, number_of_antibodies,
    measurement_time, association_constants, dissociation_constants, interference_matrix)   
ax = None
fig = None

ax, fig = outputTiterCurvePlot(
    y, log_titers, number_of_antigens, dilutions, fig=fig, ax=ax)
