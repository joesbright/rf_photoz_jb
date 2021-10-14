import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/projects/b1053/joe/photoz/crossmatching/filtered_output/test.csv')

# Select which features we wish to train on
training_features = ['unwise_w1_mag_ab', 'unwise_w2_mag_ab', # unWISE photometry
                     'gPSFMag', 'rPSFMag', 'iPSFMag', 'zPSFMag', 'yPSFMag', # panSTARRS psf photometry
                     'gKronMag', 'rKronMag', 'iKronMag', 'zKronMag', 'yKronMag', # panSTARRS Kron photometry
                     'ps1dr2_p_hsff', 'ps1dr2_p_spiral', 'ps1dr2_p_star', # metrics from Adriano
                     'gPSFMag_rPSFMag'
]

# Select the predictor that we're using to guide the model
predictor = ['z']

# Renormalise our features and the predictor
scalar = MinMaxScaler()
data[training_features + predictor] = scalar.fit_transform(data[training_features + predictor])

# Split our data set into a training and testing set in order to test our model performance
X_train, X_test, y_train, y_test = train_test_split(data[training_features], data[predictor], test_size=0.2)

# Define the random forest model
RF_reg = RandomForestRegressor(n_estimators=50, random_state=42, verbose=1)
# Fit the model with the training data
RF_reg.fit(X_train, y_train)


# Predict redshifts from the test data
test_predict = RF_reg.predict(X_test)

plt.scatter(test_predict, y_test)
plt.plot(y_test, y_test)
plt.xlabel('predictions')
plt.ylabel('real')
plt.xscale('log')
plt.yscale('log')
plt.savefig('predictions.png')
plt.clf()

plt.scatter(RF_reg.predict(X_train), y_train)
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

print('The average bias is: {}'.format(np.average((np.asarray(test_predict) - np.asarray(y_test['z']))/(1. + np.asarray(y_test['z'])))))

