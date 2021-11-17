import pandas as pd
import glob as glob
import numpy as np
import os


output_dir = '/projects/b1053/joe/photoz/crossmatching/filtered_output/'
data_input_folder = '/projects/b1053/joe/photoz/crossmatching/output/*.csv'
xmatched = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', data_input_folder))))
xmatched = xmatched.sample(frac=1, random_state=42).reset_index(drop=True)

# Only keep primary detections
filtered_xmatched = xmatched[xmatched['primaryDetection'] == 1]
# Only keep redshifts above 0 and below 1
filtered_xmatched = filtered_xmatched[filtered_xmatched['z'] > 0.]
filtered_xmatched = filtered_xmatched[filtered_xmatched['z'] < 1.]
# Filter classes
filtered_xmatched = filtered_xmatched[filtered_xmatched['class'] == 'GALAXY']
# Filter stars
filtered_xmatched = filtered_xmatched[filtered_xmatched['ps1dr2_p_star'] < 0.5]

filtered_xmatched['gmomentXX'] = filtered_xmatched['gmomentXX'].fillna(np.average(filtered_xmatched['gmomentXX'][~filtered_xmatched['gmomentXX'].isnull()]))
filtered_xmatched['gmomentXY'] = filtered_xmatched['gmomentXY'].fillna(np.average(filtered_xmatched['gmomentXY'][~filtered_xmatched['gmomentXY'].isnull()]))
filtered_xmatched['rmomentXX'] = filtered_xmatched['rmomentXX'].fillna(np.average(filtered_xmatched['rmomentXX'][~filtered_xmatched['rmomentXX'].isnull()]))
filtered_xmatched['rmomentXY'] = filtered_xmatched['rmomentXY'].fillna(np.average(filtered_xmatched['rmomentXY'][~filtered_xmatched['rmomentXY'].isnull()]))
filtered_xmatched['imomentXX'] = filtered_xmatched['imomentXX'].fillna(np.average(filtered_xmatched['imomentXX'][~filtered_xmatched['imomentXX'].isnull()]))
filtered_xmatched['imomentXY'] = filtered_xmatched['imomentXY'].fillna(np.average(filtered_xmatched['imomentXY'][~filtered_xmatched['imomentXY'].isnull()]))
filtered_xmatched['zmomentXX'] = filtered_xmatched['zmomentXX'].fillna(np.average(filtered_xmatched['zmomentXX'][~filtered_xmatched['zmomentXX'].isnull()]))
filtered_xmatched['zmomentXY'] = filtered_xmatched['zmomentXY'].fillna(np.average(filtered_xmatched['zmomentXY'][~filtered_xmatched['zmomentXY'].isnull()]))
filtered_xmatched['ymomentXX'] = filtered_xmatched['ymomentXX'].fillna(np.average(filtered_xmatched['ymomentXX'][~filtered_xmatched['ymomentXX'].isnull()]))
filtered_xmatched['ymomentXY'] = filtered_xmatched['ymomentXY'].fillna(np.average(filtered_xmatched['ymomentXY'][~filtered_xmatched['ymomentXY'].isnull()]))

filtered_xmatched['unwise_w1_mag_ab'] = filtered_xmatched['unwise_w1_mag_ab'].replace('None', np.nan)
filtered_xmatched['unwise_w1_mag_ab'] = pd.to_numeric(filtered_xmatched['unwise_w1_mag_ab'])

filtered_xmatched['unwise_w2_mag_ab'] = filtered_xmatched['unwise_w2_mag_ab'].replace('None', np.nan)
filtered_xmatched['unwise_w2_mag_ab'] = pd.to_numeric(filtered_xmatched['unwise_w2_mag_ab'])

# Fill nans with the average of the parameter
filtered_xmatched['gPSFMag'] = filtered_xmatched['gPSFMag'].fillna(np.average(filtered_xmatched['gPSFMag'][~filtered_xmatched['gPSFMag'].isnull()]))
filtered_xmatched['rPSFMag'] = filtered_xmatched['rPSFMag'].fillna(np.average(filtered_xmatched['rPSFMag'][~filtered_xmatched['rPSFMag'].isnull()]))
filtered_xmatched['iPSFMag'] = filtered_xmatched['iPSFMag'].fillna(np.average(filtered_xmatched['iPSFMag'][~filtered_xmatched['iPSFMag'].isnull()]))
filtered_xmatched['zPSFMag'] = filtered_xmatched['zPSFMag'].fillna(np.average(filtered_xmatched['zPSFMag'][~filtered_xmatched['zPSFMag'].isnull()]))
filtered_xmatched['yPSFMag'] = filtered_xmatched['yPSFMag'].fillna(np.average(filtered_xmatched['yPSFMag'][~filtered_xmatched['yPSFMag'].isnull()]))

filtered_xmatched['gKronMag'] = filtered_xmatched['gKronMag'].fillna(np.average(filtered_xmatched['gKronMag'][~filtered_xmatched['gKronMag'].isnull()]))
filtered_xmatched['rKronMag'] = filtered_xmatched['rKronMag'].fillna(np.average(filtered_xmatched['rKronMag'][~filtered_xmatched['rKronMag'].isnull()]))
filtered_xmatched['iKronMag'] = filtered_xmatched['iKronMag'].fillna(np.average(filtered_xmatched['iKronMag'][~filtered_xmatched['iKronMag'].isnull()]))
filtered_xmatched['zKronMag'] = filtered_xmatched['zKronMag'].fillna(np.average(filtered_xmatched['zKronMag'][~filtered_xmatched['zKronMag'].isnull()]))
filtered_xmatched['yKronMag'] = filtered_xmatched['yKronMag'].fillna(np.average(filtered_xmatched['yKronMag'][~filtered_xmatched['yKronMag'].isnull()]))

filtered_xmatched['unwise_w1_mag_ab'] = filtered_xmatched['unwise_w1_mag_ab'].fillna(np.average(filtered_xmatched['unwise_w1_mag_ab'][~filtered_xmatched['unwise_w1_mag_ab'].isnull()]))    
filtered_xmatched['unwise_w2_mag_ab'] = filtered_xmatched['unwise_w2_mag_ab'].fillna(np.average(filtered_xmatched['unwise_w2_mag_ab'][~filtered_xmatched['unwise_w2_mag_ab'].isnull()]))

# Add all the available colours
PSF = ['gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag']
Kron = ['gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag']
W = ['unwise_w1_mag_ab', 'unwise_w2_mag_ab']
total = PSF + Kron + W

for i, c1 in enumerate(total):
    for j, c2 in enumerate(total):
        if j > i:
            colname = c1 + '_' + c2
            filtered_xmatched[colname] = filtered_xmatched[c1] - filtered_xmatched[c2]
    

filtered_xmatched.to_csv(output_dir + 'filtered_wise_panSTARRS.csv', index=False)
