import pandas as pd
import glob as glob

# Locations of csv catalog files
unWISE_file = '/projects/b1053/joe/photoz/catalogs/unWISE/sdss_ps1dr2_unwise_33.csv'
panSTARRS_file_loc = '/projects/b1053/joe/photoz/catalogs/panSTARRS/'

# Read in csv data as pandas dataframe
unWISE_data = pd.read_csv(unWISE_file)
unWISE_data.drop_duplicates()

for panSTARRS_file in glob.glob(panSTARRS_file_loc + '*.csv'):

    panSTARRS_data = pd.read_csv(panSTARRS_file)
    panSTARRS_data.drop_duplicates()

    # Define identifier in unWISE data that we will use to match with panSTARRS data
    unWISE_ID_col = 'ps1dr2_id'

    # Define the ID in panSTARRS that will be used to match with unWISE data
    panSTARRS_ID_col = 'objid'

    # Filter down both datasets
    unWISE_reduced = unWISE_data[unWISE_data[unWISE_ID_col].isin(panSTARRS_data[panSTARRS_ID_col])]
    panSTARRS_reduced = panSTARRS_data[panSTARRS_data[panSTARRS_ID_col].isin(unWISE_data[unWISE_ID_col])]

    # Now perform the merge
    new_df = pd.merge(unWISE_reduced, panSTARRS_reduced, how='inner', left_on=unWISE_ID_col, right_on=panSTARRS_ID_col)

    boolean_series = panSTARRS_data.objid.isin(unWISE_data[unWISE_ID_col])
    print('Found {} matches in {}'.format(len(panSTARRS_data[boolean_series]), panSTARRS_file))

    if len(panSTARRS_data[boolean_series]) > 0:
        print('Writing output to {}'.format(panSTARRS_file.split('/')[-1] + '_unwise.csv'))
        new_df.to_csv('output/' + panSTARRS_file.split('/')[-1] + '_unwise.csv', index=False)
    else:
        print('No matches. Not writing.')
