#!/usr/bin/env python
# coding: utf-8

# # The Realm of Supervised Learning

# ## Preprocessing data using different techniques

# In[ ]:


import numpy as np
from sklearn import preprocessing


# In[ ]:


data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])


# ### Mean removal

# In[ ]:


data_standardized = preprocessing.scale(data)
print("Mean =", data_standardized.mean(axis=0))
print("Std deviation =", data_standardized.std(axis=0))


# ### Scaling

# In[ ]:


data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("Min max scaled data =", data_scaled)


# ### Normalization

# In[ ]:


data_normalized = preprocessing.normalize(data, norm='l1')
print("L1 normalized data =", data_normalized)


# ### Binarization

# In[ ]:


data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("Binarized data =", data_binarized)


# ### One Hot Encoding

# In[ ]:


encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("Encoded vector =", encoded_vector)


# ## Label encoding

# In[ ]:


label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)
print("Class mapping:")
for i, item in enumerate(label_encoder.classes_):
    print(item, "-->", i)


# In[ ]:


labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print("Labels =", labels)
print("Encoded labels =", list(encoded_labels))


# In[ ]:


encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("Encoded labels =", encoded_labels)
print("Decoded labels =", list(decoded_labels))


# ## Building a linear regressor

# In[ ]:


import sys
import numpy as np


# In[ ]:


filename = sys.argv[1]
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)


# In[ ]:


num_training = int(0.8 * len(X))
num_test = len(X) - num_training


# In[ ]:


# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])


# In[ ]:


from sklearn import linear_model


# In[ ]:


# Create linear regression object
linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()


# In[ ]:


y_test_pred = linear_regressor.predict(X_test)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()


# ## Computing regression accuracy

# In[ ]:


import sklearn.metrics as sm


# In[ ]:


print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))


# ## Building a ridge regressor

# In[ ]:


ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)


# In[ ]:


ridge_regressor.fit(X_train, y_train)
Y_test_pred_ridge = ridge_regressor.predict(X_test)
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge), 2))


# ## Building a polynomial regressor

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


polynomial = PolynomialFeatures(degree=3)


# In[ ]:


X_train_transformed = polynomial.fit_transform(X_train)


# In[ ]:


datapoint = [0.39,2.78,7.11]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("Linear regression:", linear_regressor.predict(datapoint)[0])
print("Polynomial regression:", poly_linear_model.predict(poly_datapoint)[0])


# ## Estimating housing prices

# In[ ]:


import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# In[ ]:


housing_data = datasets.load_boston()


# In[ ]:


housing_data.data


# In[ ]:


housing_data.target


# In[ ]:


X, y = shuffle(housing_data.data, housing_data.target, random_state=7)


# In[ ]:


num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]


# In[ ]:


dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)


# In[ ]:


ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)


# In[ ]:


y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)
r2 = r2_score(y_test, y_pred_dt)
print("#### Decision Tree performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print("R2 score =", round(r2, 2))


# In[ ]:


y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
r2 = r2_score(y_test, y_pred_ab)
print("#### AdaBoost performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print("R2 score =", round(r2, 2))


# ## Computing the relative importance of features

# In[ ]:


def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values
    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    
    # Sort the index values and flip them so that they are arranged in decreasing order of importance
    index_sorted = np.flipud(np.argsort(feature_importances))
    
    # Center the location of the labels on the X-axis (for display purposes only)
    pos = np.arange(index_sorted.shape[0]) + 0.5
    
    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


# In[ ]:


plot_feature_importances(dt_regressor.feature_importances_, 'Decision Tree regressor', housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_, 'AdaBoost regressor', housing_data.feature_names)


# ## Estimating bicycle demand distribution

# In[ ]:


import csv
from sklearn.ensemble import RandomForestRegressor


# In[ ]:


def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'rt'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[3:14])
        y.append(row[-1])
        
    # Extract feature names
    feature_names = np.array(X[0])
    
    # Remove the first row because they are feature names
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names


# In[ ]:


X, y, feature_names = load_dataset('datasets/bike/hour.csv')
X, y = shuffle(X, y, random_state=7)


# In[ ]:


num_traning = int(0.9 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]


# In[ ]:


rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2)
rf_regressor.fit(X_train, y_train)


# In[ ]:


y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("#### Random Forest regressor performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))
print("R2 score =", round(r2, 2))


# In[ ]:


plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)

