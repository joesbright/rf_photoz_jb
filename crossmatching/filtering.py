import pandas as pd
import glob as glob

input_dir = '/projects/b1053/joe/photoz/crossmatching/output/'
output_dir = '/projects/b1053/joe/photoz/crossmatching/filtered_output/'

input_files = glob.glob(input_dir + '*')

for f in input_files:
    print('Filtering: {}'.format(f))
    xmatched = pd.read_csv(f)
    
    # only keep primary detections
    filtered_xmatched = xmatched[xmatched['primaryDetection'] == 1]
    filtered_xmatched = filtered_xmatched[filtered_xmatched['z'] > 0]
    filtered_xmatched['gPSFMag'] = filtered_xmatched['gPSFMag'].fillna(-999)
    filtered_xmatched['rPSFMag'] = filtered_xmatched['rPSFMag'].fillna(-999)
    filtered_xmatched['iPSFMag'] = filtered_xmatched['iPSFMag'].fillna(-999)
    filtered_xmatched['zPSFMag'] = filtered_xmatched['zPSFMag'].fillna(-999)
    filtered_xmatched['yPSFMag'] = filtered_xmatched['yPSFMag'].fillna(-999)

    filtered_xmatched['gKronMag'] = filtered_xmatched['gKronMag'].fillna(-999)
    filtered_xmatched['rKronMag'] = filtered_xmatched['rKronMag'].fillna(-999)
    filtered_xmatched['iKronMag'] = filtered_xmatched['iKronMag'].fillna(-999)
    filtered_xmatched['zKronMag'] = filtered_xmatched['zKronMag'].fillna(-999)
    filtered_xmatched['yKronMag'] = filtered_xmatched['yKronMag'].fillna(-999)

    # add some features
    filtered_xmatched['gPSFMag_rPSFMag'] = filtered_xmatched['gPSFMag'] - filtered_xmatched['rPSFMag']

    filtered_xmatched.to_csv(output_dir + f.split('/')[-1], index=False)
