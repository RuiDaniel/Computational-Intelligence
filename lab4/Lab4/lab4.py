import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


iris = pd.read_csv('iris/iris.data', header=None, names = ['s_length', 's_width', 'p_length', 'p_width', 'class'])
print(iris.iloc[0]['class'])
print(iris.iloc[1]['class'])

sns.set_theme()
sns.relplot(
    data=iris,
    x="s_length", y="s_width", hue="class", style="class", size="p_width"
)
plt.show()

print(iris.head)
print(iris.tail)
print(iris.dtypes)

haberman = pd.read_csv('haberman/haberman.data', header=None, names = ['age', 'operation_year', 'nodes_detected', 'status'])

print(haberman.head)
print(haberman.tail)
print(haberman.dtypes)


y = (np.array(iris))[:,-1]
# print(y)

X = (np.array(iris))[:,:-1]
# print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

##########################################################################################
# GaussianNB

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

recall = recall_score(y_test, y_pred, average=None)
print('recall: {}'.format(recall))


matrix_conf = confusion_matrix(y_test, y_pred)
print('confusion matrix: \n {}'.format(matrix_conf))

precision = precision_score(y_test, y_pred, average=None)
print('precision: {}'.format(precision))

accuracy = accuracy_score(y_test, y_pred)
print('accuracy: {}'.format(accuracy))

##########################################################################################
# LinearSVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


clf2 = make_pipeline(StandardScaler(),LinearSVC(dual=False, random_state=0, tol=1e-5))

clf2.fit(X_train, y_train)

y_pred = clf2.predict(X_test)

recall = recall_score(y_test, y_pred, average=None)
print('recall: {}'.format(recall))


matrix_conf = confusion_matrix(y_test, y_pred)
print('confusion matrix: \n {}'.format(matrix_conf))

precision = precision_score(y_test, y_pred, average=None)
print('precision: {}'.format(precision))

accuracy = accuracy_score(y_test, y_pred)
print('accuracy: {}'.format(accuracy))

##########################################################################################
# svm.SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf3 = svm.SVC()
clf3.fit(X_train, y_train)

y_pred = clf3.predict(X_test)


recall = recall_score(y_test, y_pred, average=None)
print('recall: {}'.format(recall))


matrix_conf = confusion_matrix(y_test, y_pred)
print('confusion matrix: \n {}'.format(matrix_conf))

precision = precision_score(y_test, y_pred, average=None)
print('precision: {}'.format(precision))

accuracy = accuracy_score(y_test, y_pred)
print('accuracy: {}'.format(accuracy))



##########################################################################################
# KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)


recall = recall_score(y_test, y_pred, average=None)
print('recall: {}'.format(recall))


matrix_conf = confusion_matrix(y_test, y_pred)
print('confusion matrix: \n {}'.format(matrix_conf))

precision = precision_score(y_test, y_pred, average=None)
print('precision: {}'.format(precision))

accuracy = accuracy_score(y_test, y_pred)
print('accuracy: {}'.format(accuracy))