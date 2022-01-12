#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:46:18 2022

@author: jsb2081
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def label(xlab, ylab):
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return
    

unWISE_loc = '/projects/b1053/joe/photoz/catalogs/unWISE/'
unWISE_file = 'sdss_ps1dr2_unwise_33.csv'

mycols = ['spec_obj_id',
          'target_obj_id',
          'objid',
          'sdss_ra',
          'sdss_decl',
          'photo_ra',
          'photo_decl',
          'z',
          'z_err',
          'class',
          'subclass',
          'z_warning',
          'type',
          'surbey',
          'ps1dr2_angular_separation',
          'ps1dr2_decl',
          'ps1dr2_id',
          ]

data = pd.read_csv(unWISE_loc + unWISE_file)

cols = list(data)

mycols = ['z',
          'sdss_ra',
          'sdss_decl',
          'ps1dr2_angular_separation']

for col in mycols:
    plt.hist(x=data[col], bins=40, log=True)
    label(col, 'counts')
    plt.savefig(os.getcwd() + '/hists/' + str(col) + '_hist.png')
    plt.clf()
    
mycategoric_cols = ['type',
                     'class',
                     'subclass']

for col in mycategoric_cols:
    print(col + ' contains ' + str(set(data[col])))



