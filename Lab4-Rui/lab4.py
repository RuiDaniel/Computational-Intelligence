import numpy as np
import pandas as pd

# Column names to be added
column_names_haberman = ["Time", "Year", "Num Pos Nodes", "Survival"]

hab = pd.read_csv('haberman/haberman/haberman.data', names=column_names_haberman)

# Column names to be added 
column_names_iris = ["s_length", "s_width", "p_length", "p_width", "class"]

iris = pd.read_csv('iris/iris/iris.data', names=column_names_iris)
bez = pd.read_csv('iris/iris/bezdekIris.data', names=column_names_iris)

# print(hab.head(1))
# print(hab.tail(1))
# print(hab.dtypes)

# print(iris.head(1))
# print(iris.tail(1))
# print(iris.dtypes)

# print(bez.head(1))
# print(bez.tail(1))
# print(bez.dtypes)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
sns.relplot(
    data=iris,
    x="s_length", y="s_width", hue="class", style="class", size="p_width"
)
plt.show()

# sns.set_theme()
# sns.relplot(
#     data=bez,
#     x="s_length", y="s_width", hue="class", style="class", size="p_width"
# )
# plt.show()

# sns.set_theme()
# sns.relplot(
#     data=hab,
#     x="Time", y="Year", hue="Survival", style="Survival", size="Num Pos Nodes"
# )
# plt.show()


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

y = (np.array(iris))[:,-1]
# print(y)

X = (np.array(iris))[:,:-1]
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

def GaussianNB_method(X_train, X_test, y_train):
    clf = GaussianNB()
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    return y_pred

def LinearSVC_method(X_train, X_test, y_train):
    clf = make_pipeline(StandardScaler(), LinearSVC(dual="auto", random_state=0, tol=1e-5))
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    return y_pred

def SVM_method(X_train, X_test, y_train):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    return y_pred

def K_Neighbors_method(X_train, X_test, y_train):
    clf = KNeighborsClassifier(n_neighbors=3)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    return y_pred

y_pred = GaussianNB_method(X_train, X_test, y_train)
# y_pred = LinearSVC_method(X_train, X_test, y_train)
# y_pred = SVM_method(X_train, X_test, y_train)
# y_pred = K_Neighbors_method(X_train, X_test, y_train)

recall = recall_score(y_test, y_pred, average=None)
print('recall: {}'.format(recall))

matrix_conf = confusion_matrix(y_test, y_pred)
print('confusion matrix: \n {}'.format(matrix_conf))

precision = precision_score(y_test, y_pred, average=None)
print('precision: {}'.format(precision))

accuracy = accuracy_score(y_test, y_pred)
print('accuracy: {}'.format(accuracy))