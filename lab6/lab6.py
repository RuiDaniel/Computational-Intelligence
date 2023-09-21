import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV  # also does cross validation for us
from sklearn.pipeline import Pipeline


iris = pd.read_csv('lab6/iris/iris.data', header=None, names = ['s_length', 's_width', 'p_length', 'p_width', 'class'])

# print(iris.head)
# print(iris.tail)
print(iris.dtypes)

haberman = pd.read_csv('lab6/haberman/haberman.data', header=None, names = ['age', 'operation_year', 'nodes_detected', 'status'])

# print(haberman.head)
# print(haberman.tail)
print(haberman.dtypes)


y = (np.array(iris))[:,-1]
# print(y)

X = (np.array(iris))[:,:-1]
# print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

y_haber = (np.array(haberman))[:,-1]
# print(y)

X_haber = (np.array(haberman))[:,:-1]
# print(X)

X_train_haber, X_test_haber, y_train_haber, y_test_haber = train_test_split(X_haber, y_haber, test_size=0.3, random_state=42)




###################################################################################################
### Iris DATASET ###


mlp = MLPClassifier(max_iter=100)

# Create a pipeline with StandardScaler and MLPClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normalization step
    ('classifier', mlp)  # MLPClassifier step
])

test_params = {
    'classifier__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (20, 20, 20, 20)],
    'classifier__activation': ['tanh', 'relu'],
    'classifier__solver': ['sgd', 'adam', 'lbfgs'],
    'classifier__alpha': [0.0001, 0.001, 0.005, 0.01, 0.05],
    'classifier__learning_rate': ['constant', 'adaptive'],
}

grid_search = GridSearchCV(pipeline, test_params, cv=5)
grid_search.fit(X_train, y_train)

best_mlp = grid_search.best_estimator_

y_pred = best_mlp.predict(X_test)

###################################################################################################
### haberman DATASET ###

mlp_haberman = MLPClassifier(max_iter=100)

# Create a pipeline with StandardScaler and MLPClassifier
pipeline_haberman = Pipeline([
    ('scaler', StandardScaler()),  # Normalization step
    ('classifier_haberman', mlp_haberman)  # MLPClassifier step
])

test_params_haberman = {
    'classifier_haberman__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (20, 20, 20, 20)],
    'classifier_haberman__activation': ['tanh', 'relu'],
    'classifier_haberman__solver': ['sgd', 'adam', 'lbfgs'],
    'classifier_haberman__alpha': [0.0001, 0.001, 0.005, 0.01, 0.05],
    'classifier_haberman__learning_rate': ['constant', 'adaptive'],
}

grid_search_haberman = GridSearchCV(pipeline_haberman, test_params_haberman, cv=5)
grid_search_haberman.fit(X_train_haber, y_train_haber)

print(grid_search_haberman.best_params_)

best_mlp_haberman = grid_search_haberman.best_estimator_

y_pred_haber = best_mlp_haberman.predict(X_test_haber)



######################################################################################################
### Evaluation ###

recall_haber = recall_score(y_test_haber, y_pred_haber, average=None)
print('recall_haber: {}'.format(recall_haber))


matrix_conf_haber = confusion_matrix(y_test_haber, y_pred_haber)
print('confusion matrix_haber: \n {}'.format(matrix_conf_haber))

precision_haber = precision_score(y_test_haber, y_pred_haber, average=None)
print('precision_haber: {}'.format(precision_haber))

accuracy_haber = accuracy_score(y_test_haber, y_pred_haber)
print('accuracy_haber: {}'.format(accuracy_haber))


print(grid_search.best_params_)

recall = recall_score(y_test, y_pred, average=None)
print('recall: {}'.format(recall))


matrix_conf = confusion_matrix(y_test, y_pred)
print('confusion matrix: \n {}'.format(matrix_conf))

precision = precision_score(y_test, y_pred, average=None)
print('precision: {}'.format(precision))

accuracy = accuracy_score(y_test, y_pred)
print('accuracy: {}'.format(accuracy))