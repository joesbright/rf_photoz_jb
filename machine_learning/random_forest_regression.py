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
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Select which features we wish to train on
training_features = [#'unwise_w1_mag_ab', 'unwise_w2_mag_ab', # unWISE photometry
                     'gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag', # panSTARRS psf photometry
                     'gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag', # panSTARRS Kron photometry
                     ]

PSF = ['gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag']
Kron = ['gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag']
#W = []
total = PSF + Kron

for i, c1 in enumerate(total):
    for j, c2 in enumerate(total):
        if j > i:
            colname = c1 + '_' + c2
            training_features.append(colname)


# Select the predictor that we're using to guide the model
predictor = ['z']

# Renormalise our features and the predictor
scaler = StandardScaler()

data[training_features + predictor] = scaler.fit_transform(data[training_features + predictor])

z_mean = scaler.mean_[-1]
z_std = np.sqrt(scaler.var_[-1])


# Split our data set into a training and testing set in order to test our model performance
X_train, X_test, y_train, y_test = train_test_split(data[training_features], data[predictor], test_size=0.2)

# Define the random forest model
RF_reg = RandomForestRegressor(n_estimators=500, random_state=42, verbose=2, n_jobs=-1)
# Fit the model with the training data
RF_reg.fit(X_train, y_train)


# Predict redshifts from the test data
test_predict = RF_reg.predict(X_test)

test_predict = test_predict * z_std + z_mean
y_test = y_test * z_std + z_mean
y_train = y_train * z_std + z_mean

plt.scatter(test_predict, y_test)
plt.plot(y_test, y_test)
plt.xlabel('predictions')
plt.ylabel('real')
plt.xscale('log')
plt.yscale('log')
plt.savefig('predictions.png')
plt.clf()

plt.scatter(RF_reg.predict(X_train) * z_std + z_mean, y_train)
plt.plot(y_train, y_train)
plt.xlabel('predictions')
plt.ylabel('real')
plt.xscale('log')
plt.yscale('log')
plt.savefig('model.png')
plt.clf()

importances = RF_reg.feature_importances_
std = np.std([tree.feature_importances_ for tree in RF_reg.estimators_], axis=0)
forest_importances = pd.Series(importances, index=training_features)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig('importance.png')
plt.clf()

plt.scatter(y_test['z'], (np.asarray(test_predict) - np.asarray(y_test['z']))/(1. + np.asarray(y_test['z'])))
plt.savefig('norm.png')
plt.clf()

plt.barh(training_features, RF_reg.feature_importances_)
plt.savefig('importances2.png')

print('The average bias is: {}'.format(np.average((np.asarray(test_predict) - np.asarray(y_test['z']))/(1. + np.asarray(y_test['z'])))))


joblib.dump(RF_reg, './models/test_model.joblib')
