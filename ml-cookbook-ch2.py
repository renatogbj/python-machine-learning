#!/usr/bin/env python
# coding: utf-8

# # Constructing a Classifier

# ## Building a simple classifier

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


X = np.array([[3,1], [2,5], [1,8], [6,4], [5,2], [3,5], [4,7], [4,-1]])
y = [0, 1, 1, 0, 0, 1, 1, 0]


# In[ ]:


class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])


# In[ ]:


plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')


# In[ ]:


line_x = range(10)
line_y = line_x


# In[ ]:


plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')
plt.plot(line_x, line_y, color='black', linewidth=3)
plt.show()


# ## Building a logistic regression classifier

# In[ ]:


from sklearn import linear_model


# In[ ]:


X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


# In[ ]:


classifier = linear_model.LogisticRegression(solver='liblinear', C=100)


# In[ ]:


classifier.fit(X, y)


# In[ ]:


def plot_classifier(classifier, X, y):
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
    
    # plot the output using a colored plot
    plt.figure()
    
    # choose a color scheme
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    
    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    
    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))
    
    plt.show()


# In[ ]:


plot_classifier(classifier, X, y)


# In[ ]:


classifier = linear_model.LogisticRegression(solver='liblinear', C=1.0)
classifier.fit(X, y)
plot_classifier(classifier, X, y)


# In[ ]:


classifier = linear_model.LogisticRegression(solver='liblinear', C=10000)
classifier.fit(X, y)
plot_classifier(classifier, X, y)


# ## Building a Naive Bayes classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


input_file = 'datasets/data_multivar.txt'

X, y = [], []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])
        
X = np.array(X)
y = np.array(y)


# In[ ]:


classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)


# In[ ]:


accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")


# In[ ]:


plot_classifier(classifier_gaussiannb, X, y)


# ## Splitting the dataset for training and testing

# In[ ]:


from sklearn import model_selection


# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb_new = GaussianNB()
classifier_gaussiannb_new.fit(X_train, y_train)


# In[ ]:


y_test_pred = classifier_gaussiannb_new.predict(X_test)


# In[ ]:


accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the classifier =", round(accuracy, 2), "%")


# In[ ]:


plot_classifier(classifier_gaussiannb_new, X_test, y_test)


# ## Evaluating the accuracy using cross-validation

# In[ ]:


num_validations = 5
accuracy = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='accuracy', cv=num_validations)
print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")


# In[ ]:


f1 = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=num_validations)
print("F1: " + str(round(100*f1.mean(), 2)) + "%")


# In[ ]:


precision = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='precision_weighted', cv=num_validations)
print("Precision: " + str(round(100*precision.mean(), 2)) + "%")


# In[ ]:


recall = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='recall_weighted')


# ## Visualizing the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


# Show confusion matrix
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Paired)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# In[ ]:


y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
confusion_mat = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(confusion_mat)


# ## Extracting the performance report

# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


y_true = [1, 0, 0, 2, 1, 0, 3, 3, 3]
y_pred = [1, 1, 0, 2, 1, 0, 1, 3, 3]
target_names = ['Class-0', 'Class-1', 'Class-2', 'Class-3']
print(classification_report(y_true, y_pred, target_names=target_names))


# ## Evaluating cars based on their characteristics

# In[ ]:


from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


input_file = 'datasets/car.data'

# Reading the data
X = []
count = 0

with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)

X = np.array(X)


# In[ ]:


# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)


# In[ ]:


# Build a Random Forest classifier
params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)


# In[ ]:


accuracy = model_selection.cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print("Accuracy of the classifier: ", str(round(100*accuracy.mean(), 2)) + "%")


# ## Extracting validation curves

# In[ ]:


# Validation curves
from sklearn.model_selection import validation_curve


# In[ ]:


classifier = RandomForestClassifier(max_depth=4, random_state=7)
parameter_grid = np.linspace(25, 200, 8).astype(int)
train_scores, validation_scores = validation_curve(classifier, X, y, "n_estimators", parameter_grid, cv=5)
print("##### VALIDATION CURVES #####")
print("Param: n_estimators\nTraining scores:\n", train_scores)
print("Param: n_estimators\nValidation scores:\n", validation_scores)


# In[ ]:


# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


classifier = RandomForestClassifier(n_estimators=20, random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int)
train_scores, valid_scores = validation_curve(classifier, X, y, "max_depth", parameter_grid, cv=5)
print("Param: max_depth\nTraining scores:\n", train_scores)
print("Param: max_depth\nValidation scores:\n", validation_scores)


# In[ ]:


plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Validation curve')
plt.xlabel('Maximum depth of the tree')
plt.ylabel('Accuracy')
plt.show()


# ## Extracting learning curves

# In[ ]:


from sklearn.model_selection import learning_curve


# In[ ]:


classifier = RandomForestClassifier(random_state=7)

parameter_grid = np.array([200, 500, 800, 1100])
train_sizes, train_scores, validation_scores = learning_curve(classifier, X, y, train_sizes=parameter_grid, cv=5)
print("##### LEARNING CURVES #####")
print("Training scores:\n", train_scores)
print("Validation scores:\n", validation_scores)


# In[ ]:


# Plot the curve
plt.figure()
plt.plot(parameter_grid, 100*np.average(train_scores, axis=1), color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()


# ## Estimating the income bracket

# In[ ]:


input_file = 'datasets/adult.data.txt'


# In[ ]:


X = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_images_threshold = 10000


# In[ ]:


with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:
            continue
        
        data = line[:-1].split(', ')
        
        if data[-1] == '<=50K' and count_lessthan50k < num_images_threshold:
            X.append(data)
            count_lessthan50k = count_lessthan50k + 1
        elif data[-1] == '>50k' and count_morethan50k < num_images_threshold:
            X.append(data)
            count_morethan50k = count_morethan50k + 1
        
        if count_lessthan50k >= num_images_threshold and count_morethan50k >= num_images_threshold:
            break

X = np.array(X)


# In[ ]:


# Convert string data to numerical data
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


classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)


# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb.predict(X_test)


# In[ ]:


f1 = model_selection.cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=5)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

