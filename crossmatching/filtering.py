import pandas as pd
import glob as glob
import numpy as np
import glob as glob


input_dir = '/projects/b1053/joe/photoz/crossmatching/output/'
output_dir = '/projects/b1053/joe/photoz/crossmatching/filtered_output/'

input_files = glob.glob(input_dir + '*')

for f in input_files:
    print('Filtering: {}'.format(f))
    xmatched = pd.read_csv(f)
    
    # Only keep primary detections
    filtered_xmatched = xmatched[xmatched['primaryDetection'] == 1]
    # Only keep redshifts above 0
    filtered_xmatched = filtered_xmatched[filtered_xmatched['z'] > 0]
    # Filter classes
    filtered_xmatched = filtered_xmatched[filtered_xmatched['class'] == 'GALAXY']
    # Filter stars
    filtered_xmatched = filtered_xmatched[filtered_xmatched['ps1dr2_p_star'] < 0.5]
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

    #filtered_xmatched['unwise_w1_mag_ab'] = filtered_xmatched['unwise_w1_mag_ab'].fillna(np.average(filtered_xmatched['unwise_w1_mag_ab'][~filtered_xmatched['unwise_w1_mag_ab'].isnull()]))

    #filtered_xmatched['unwise_w2_mag_ab'] = filtered_xmatched['unwise_w2_mag_ab'].fillna(np.average(filtered_xmatched['unwise_w2_mag_ab'][~filtered_xmatched['unwise_w2_mag_ab'].isnull()]))

    # Add all the available colours
    PSF = ['gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag']
    Kron = ['gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag']
    #W = []
    total = PSF + Kron 

    for i, c1 in enumerate(total):
        for j, c2 in enumerate(total):
            if j > i:
                colname = c1 + '_' + c2
                filtered_xmatched[colname] = filtered_xmatched[c1] - filtered_xmatched[c2]
    

    filtered_xmatched.to_csv(output_dir + f.split('/')[-1], index=False)
