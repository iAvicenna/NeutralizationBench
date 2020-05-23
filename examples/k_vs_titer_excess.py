#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:41:57 2020

@author: Sina

testing the relation between forward rates k in the case there is excess sera

"""
import pylab
import numpy as np    
import sys
sys.path.append("../")
from NeutralizationBench import titrateAntigensAgainstSera
from NeutralizationBench import outputScatterPlot


number_of_antigens = 1
number_of_antibodies = 1
M = 6e23
vratio = 50
nspike = 450
ratio = vratio * nspike
total_volume = 1e-4


total_PFU = 100
total_antibody = 5e13
ks = np.linspace(np.log2(0.01), np.log2(40), 200)
len1 = ks.size

ax = None
fig = None
titers = []
log_titers = []
for forward_rate in ks:
    association_rates = np.ones((number_of_antigens, number_of_antibodies)) * 2**forward_rate * 1e5 / M
    dissociation_rates = 1e-4 * np.ones((number_of_antigens, number_of_antibodies))
    interference_matrix = np.ones((number_of_antigens, number_of_antibodies, number_of_antibodies))

    init_vals = [total_PFU * ratio / total_volume, total_antibody / total_volume]
    
    measurement_time = 3600
    dilutions = 1 / np.array([5120, 2560, 1280, 640, 320, 160, 80, 40, 20])    
    
    y, log_titer, titer, _ = titrateAntigensAgainstSera(
        init_vals, dilutions, number_of_antigens, number_of_antibodies,
        measurement_time, association_rates, dissociation_rates, interference_matrix)   

    if '<' in str(titer[0]): 
        titer_val = int(float(titer[0][1:]) / 2)
    elif '>' in str(titer[0]):   
        titer_val = int(float(titer[0][1:]) * 2)
    else:
        titer_val = int(titer[0])

    titers.append(titer_val)
    log_titers.append(-1 * log_titer[0])        
        

ax, fig = outputScatterPlot(ks, log_titers, xlabel='k*(Me-6)', ylabel='Titer',
                            yticks=[log_titers[x] for x in range(0, len1, 20)],
                            ylabels=[titers[x] for x in range(0, len1, 20)])

        
F = pylab.gcf()
F.savefig('./figs/f_vs_titer_excess.png', dpi=300) 
