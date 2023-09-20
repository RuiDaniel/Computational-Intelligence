import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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