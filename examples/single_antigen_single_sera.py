#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:41:57 2020

@author: Sina


test single antigen single sera
"""
import sys
sys.path.append("../")
from NeutralizationBench import titrateAntigensAgainstSera
from NeutralizationBench import outputTiterCurvePlot

import numpy as np    

number_of_antigens=1
number_of_antibodies=1
M=6e23
vratio=50
nspike=450
ratio=vratio*nspike
total_volume=1e-4

forward_rates=np.ones((number_of_antigens,number_of_antibodies))*60e9/(M)
backward_ratios=1e-3*np.ones((number_of_antigens,number_of_antibodies))
interference_matrix=np.ones((number_of_antigens,number_of_antibodies,number_of_antibodies))


total_PFU=1000
total_antibody=10*1e9
init_vals=[total_PFU*ratio/total_volume,total_antibody/total_volume]


measurement_time=3600
dilutions=[1/5120,1/2560,1/1280,1/640,1/320,1/160,1/80,1/40,1/20]

y,log_titers,titers,_=titrateAntigensAgainstSera(init_vals,dilutions,number_of_antigens,number_of_antibodies,measurement_time,forward_rates,backward_ratios,interference_matrix)   

ax,fig=outputTiterCurvePlot(y,log_titers,number_of_antigens,dilutions,fig=None,ax=None)
