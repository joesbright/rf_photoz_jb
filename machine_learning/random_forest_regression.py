import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib as joblib
import glob as glob
import os as os


data_input_folder = '/projects/b1053/joe/photoz/crossmatching/filtered_output/*.csv'
data = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', data_input_folder))))
#data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Select which features we wish to train on
training_features = ['unwise_w1_mag_ab', 'unwise_w2_mag_ab', # unWISE photometry
                     'gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag', # panSTARRS psf photometry
                     'gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag', # panSTARRS Kron photometry
                     'gmomentXX', 'gmomentXY',
                     'rmomentXX', 'rmomentXY',
                     'imomentXX', 'imomentXY',
                     'zmomentXX', 'zmomentXY',
                     'ymomentXX', 'ymomentXY',
                     ]

# Add colors to feature list
PSF = ['gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag']
Kron = ['gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag']
W = ['unwise_w1_mag_ab', 'unwise_w2_mag_ab']
total = PSF + Kron + W

for i, c1 in enumerate(total):
    for j, c2 in enumerate(total):
        if j > i:
            colname = c1 + '_' + c2
            training_features.append(colname)

# Select the predictor that we're using to guide the model
predictor = 'z'

# Renormalise our features and the predictor
scaler = StandardScaler()

data[training_features + list(predictor)] = scaler.fit_transform(data[training_features + list(predictor)])

z_mean = scaler.mean_[-1]
z_std = np.sqrt(scaler.var_[-1])

# Split our data set into a training and testing set in order to test our model performance
X_train, X_test, y_train, y_test = train_test_split(data[training_features], data[predictor], test_size=0.8, random_state=42)

print(X_train)

# Define the random forest model
RF_reg = RandomForestRegressor(n_estimators=500, random_state=42, verbose=2, n_jobs=-1)
# Fit the model with the training data
RF_reg.fit(X_train, y_train)

# Predict redshifts from the test data
test_predict = RF_reg.predict(X_test) * z_std + z_mean
train_predict = RF_reg.predict(X_train) * z_std + z_mean
y_train = y_train * z_std + z_mean
y_test = y_test * z_std + z_mean

# Compute the bias
bias = (np.asarray(test_predict) - np.asarray(y_test))/(1. + np.asarray(y_test))
bias_std = np.std(bias)
filtered_bias = []
for item in bias:
    if np.abs(item) < 3. * bias_std:
        filtered_bias.append(item)
filtered_bias = np.asarray(filtered_bias)
print('The average bias is: {}'.format(np.average(bias)))
print('The bias with outliars removed is: {}'.format(np.average(filtered_bias)))
print('The outlier percentage is: {}%'.format(100.*(len(bias)-len(filtered_bias))/len(bias)))

plt.scatter(y_train, train_predict)
plt.plot(y_train, y_train)
plt.xlabel('Zspec')
plt.ylabel('Zphot')
plt.savefig('test.png')
plt.clf()

plt.scatter(y_test, test_predict)
plt.plot(y_test, y_test)
plt.xlabel('Zspec')
plt.ylabel('Zphot')
plt.savefig('test2.png')
plt.clf()

plt.scatter(y_test, bias)
plt.axhline(-3. * bias_std)
plt.axhline(3. * bias_std)
plt.xlabel('Zspec')
plt.ylabel('Bias')
plt.savefig('bias.png')

joblib.dump(RF_reg, './models/test_model.joblib')
