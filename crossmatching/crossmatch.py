import pandas as pd

# Locations of csv catalog files
unWISE_file = '/projects/b1053/joe/photoz/catalogs/unWISE/sdss_galaxies_fitered_unwise.csv'
panSTARRS_file = '/projects/b1053/joe/photoz/catalogs/panSTARRS/SOV_10.10.db.csv'

# Read in csv data as pandas dataframe
unWISE_data = pd.read_csv(unWISE_file)
unWISE_data.drop_duplicates()
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

new_df.to_csv('output/test.csv', index=False)

print('matching IDs:')
boolean_series = panSTARRS_data.objid.isin(unWISE_data[unWISE_ID_col])
print(len(panSTARRS_data[boolean_series]))

