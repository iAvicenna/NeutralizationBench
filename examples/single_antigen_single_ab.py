#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:41:57 2020

@author: Sina


test single antigen single sera
"""
import sys
import numpy as np    
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

association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 1e5 / M
dissociation_rates = 1e-3 * np.ones((number_of_antigens, number_of_antibodies))
interference_matrix = np.ones((number_of_antigens, number_of_antibodies,
                               number_of_antibodies))


total_PFU = 1000
total_antibody = 5e13
init_vals = [total_PFU * ratio / total_volume, total_antibody / total_volume]


measurement_time = 3600
dilutions = 1 / np.array([5120, 2560, 1280, 640, 320, 160, 80, 40, 20])

y, log_titers, titers, _ = titrateAntigensAgainstSera(
    init_vals, dilutions, number_of_antigens, number_of_antibodies,
    measurement_time, association_rates, dissociation_rates, interference_matrix)   

ax, fig = outputTiterCurvePlot(
    y, log_titers, number_of_antigens, dilutions, fig=None, ax=None)
