#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:41:57 2020

@author: Sina

testing one antigens single sera when there is complete interference
"""
import sys
sys.path.append("../")
from NeutralizationBench import titrateAntigensAgainstSera
from NeutralizationBench import outputTiterCurvePlot

import numpy as np    

number_of_antigens=1
number_of_antibodies=2
M=6e23
vratio=50
nspike=450
ratio=vratio*nspike
total_volume=1e-4

forward_rates=np.ones((number_of_antigens,number_of_antibodies))*60e9/(M)
total_antibody=20e9
forward_rates[0,1]*=5

backward_ratios=1e-3*np.ones((number_of_antigens,number_of_antibodies))
interference_matrix=np.ones((number_of_antigens,number_of_antibodies,number_of_antibodies))

interference_matrix[0,0,1]=0
interference_matrix[0,1,0]=0

total_PFU=1000
init_vals=[total_PFU*ratio/total_volume,0.5*total_antibody/total_volume,0.5*total_antibody/total_volume]


measurement_time=3600
dilutions=[1/5120,1/2560,1/1280,1/640,1/320,1/160,1/80,1/40,1/20]

y,log_titer,titer,_=titrateAntigensAgainstSera(init_vals,dilutions,number_of_antigens,number_of_antibodies,measurement_time,forward_rates,backward_ratios,interference_matrix)   
ax=None
fig=None
ax,fig=outputTiterCurvePlot(y,log_titer,number_of_antigens,dilutions,fig=fig,ax=ax)
