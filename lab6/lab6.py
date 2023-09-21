import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV  # also does cross validation for us


iris = pd.read_csv('iris/iris.data', header=None, names = ['s_length', 's_width', 'p_length', 'p_width', 'class'])

# print(iris.head)
# print(iris.tail)
print(iris.dtypes)

haberman = pd.read_csv('haberman/haberman.data', header=None, names = ['age', 'operation_year', 'nodes_detected', 'status'])

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


# Cross-Validation is necessary since we need to test the best number os hidden layes, density and best hyper paremeters

# make_pipeline(StandardScaler(), clf)
#         clf = make_pipeline(StandardScaler(), clf)
#         clf.fit(X_train, y_train)
#         score = clf.score(X_test, y_test)

# cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# scores = cross_val_score(clf, X, y, cv=cv)
# scores.mean()

mlp = make_pipeline(StandardScaler(), MLPClassifier(max_iter=200))

test_params = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,), (20, 20, 20, 20)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam', 'lbfgs'],
    'alpha': [0.0001, 0.001, 0.005, 0.01, 0.05],
    'learning_rate': ['constant','adaptive'],
}

grid_search = GridSearchCV(mlp, test_params, cv=5)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
