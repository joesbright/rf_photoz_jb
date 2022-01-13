import pandas as pd
import numpy as np


def find_missing(dataframe):
    '''
    A function to find the missing, nan, inf, etc values in a data frame
    '''
    for (col_name, col_data) in dataframe.iteritems():
        print('Searching ' + str(col_name) + ' for missing and bad values')
        check_for_nan = col_data.isnull()
        print('Found ' + str(check_for_nan.sum()) + ' nan values')
        check_for_inf = (col_data == np.inf) | (col_data == -np.inf)
        print('Found ' + str(check_for_inf.sum()) + ' inf values')
        check_for_str = (col_data == 'nan') | (col_data == 'NaN') | (col_data == 'none')
        print('Found ' + str(check_for_inf.sum()) + ' nan string values')
