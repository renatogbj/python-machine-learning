#!/usr/bin/env python
# coding: utf-8

# # Predictive Modeling

# ## Building a linear classifier using Support Vector Machine (SVMs)

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def load_data(input_file):
    X = []
    y = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# In[ ]:


input_file = 'datasets/data_multivar_ch3.txt'
X, y = load_data(input_file)


# In[ ]:


class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])


# In[ ]:


plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()


# In[ ]:


from sklearn import model_selection
from sklearn.svm import SVC


# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)


# In[ ]:


params = {'kernel': 'linear'}
classifier = SVC(**params)


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


# Plot the classifier boundaries on input data
def plot_classifier(classifier, X, y, title='Classifier boundaries', annotate=False):
    # define ranges to plot the figure 
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot 
    plt.figure()

    # Set the title
    plt.title(title)

    # choose a color scheme you can find all the options 
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot 
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks(())
    plt.yticks(())

    if annotate:
        for x, y in zip(X[:, 0], X[:, 1]):
            # Full documentation of the function available here: 
            # http://matplotlib.org/api/text_api.html#matplotlib.text.Annotation
            plt.annotate(
                '(' + str(round(x, 1)) + ',' + str(round(y, 1)) + ')',
                xy = (x, y), xytext = (-15, 15), 
                textcoords = 'offset points', 
                horizontalalignment = 'right', 
                verticalalignment = 'bottom', 
                bbox = dict(boxstyle = 'round,pad=0.6', fc = 'white', alpha = 0.8),
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))


# In[ ]:


plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()


# In[ ]:


y_test_pred = classifier.predict(X_test)
plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


target_names = ['Class-' + str(int(i)) for i in set(y)]
print("#"*60)
print("Classifier performance on training dataset")
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print("#"*60)


# In[ ]:


print("#"*60)
print("Classification report on test dataset")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("#"*60)


# ## Building a nonlinear classifier using SVMs

# In[ ]:


params = {'kernel': 'poly', 'degree': 3}
classifier = SVC(**params)


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


y_test_pred = classifier.predict(X_test)
plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()


# In[ ]:


print("#"*60)
print("Classification report on test dataset")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("#"*60)


# In[ ]:


params = {'kernel': 'rbf'}
classifier = SVC(**params)


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


y_test_pred = classifier.predict(X_test)
plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()


# In[ ]:


print("#"*60)
print("Classification report on test dataset")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("#"*60)


# ## Tackling class imbalance

# In[ ]:


input_file = 'datasets/data_multivar_imbalance.txt'
X, y = load_data(input_file)


# In[ ]:


class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])


# In[ ]:


plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)


# In[ ]:


params = {'kernel': 'linear'}
classifier = SVC(**params)


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


y_test_pred = classifier.predict(X_test)
plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()


# In[ ]:


print("#"*60)
print("Classification report on test dataset")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("#"*60)


# In[ ]:


params = {'kernel': 'linear', 'class_weight': 'balanced'}
classifier = SVC(**params)


# In[ ]:


classifier.fit(X_train, y_train)


# In[ ]:


y_test_pred = classifier.predict(X_test)
plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()


# In[ ]:


print("#"*60)
print("Classification report on test dataset")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("#"*60)


# ## Finding optimal hyperparameters

# In[ ]:


input_file = 'datasets/data_multivar_ch3.txt'
X, y = load_data(input_file)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)


# In[ ]:


parameter_grid = [{'kernel': ['linear'], 'C': [1, 10, 50, 600]},
                  {'kernel': ['poly'], 'degree': [2, 3]},
                  {'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 10, 50, 600]}]


# In[ ]:


metrics = ['precision', 'recall_weighted']
for metric in metrics:
    print("#### Searching optimal hyperparameters for", metric)
    classifier = model_selection.GridSearchCV(SVC(C=1), parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)
    
    means = classifier.cv_results_['mean_test_score']
    for mean, params in zip(means, classifier.cv_results_['params']):
        print(params, '-->', round(mean, 3))
        
    print("\nHighest scoring parameter set:", classifier.best_params_)


# In[ ]:


y_true, y_pred = y_test, classifier.predict(X_test)
print("\nFull performance report:\n")
print(classification_report(y_true, y_pred))


# ## Building an event predictor

# In[ ]:


from sklearn import preprocessing


# In[ ]:


input_file = 'datasets/calit2/building_event_binary.txt'
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append([data[0]] + data[2:])

X = np.array(X)


# In[ ]:


label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
        
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)


# In[ ]:


params = {'kernel': 'rbf', 'probability': True, 'class_weight': 'balanced'}
classifier = SVC(**params)
classifier.fit(X, y)


# In[ ]:


accuracy = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print("Accuracy of the classifier: " + str(round(100*accuracy.mean(), 2)) + "%")


# ## Estimating traffic

# In[ ]:


from sklearn.svm import SVR


# In[ ]:


input_file = 'datasets/traffic_data.txt'


# In[ ]:


X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
        
X = np.array(X)


# In[ ]:


label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
        
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)


# In[ ]:


params = {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2}
regressor = SVR(**params)
regressor.fit(X, y)


# In[ ]:


import sklearn.metrics as sm


# In[ ]:


y_pred = regressor.predict(X)
print("Mean absolute error =", round(sm.mean_absolute_error(y, y_pred), 2))

