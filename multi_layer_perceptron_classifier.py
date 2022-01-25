import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier

df3 = pd.read_csv('processed-data/BSAS-data/train.csv')

# Normalize the Dataset
X = np.array(df3.drop(['target', 'zip code'], axis=1))
X = preprocessing.scale(X, axis=1)
# Separate the correct answer y from the Dataset
y = np.array(df3['target'])

iterations = RepeatedKFold(n_splits=5, n_repeats=10, random_state=100)
neural_class = MLPClassifier(hidden_layer_sizes=(4, 4), activation='logistic', solver='sgd', alpha=1e-5, learning_rate='constant', random_state=100)

precision = np.array([])

for train_set, test_set in tqdm_notebook(iterations.split(X)):
    X_train, X_test = X[train_set], X[test_set]
    y_train, y_test = y[train_set], y[test_set]
    neural_class.fit(X_train, y_train)
    precision = np.append(precision, [neural_class.score(X_test, y_test)])

print(np.amax(precision))
print(np.amin(precision))

folds = len(neural_class.coefs_)

for i in range(folds):
    array = neural_class.coefs_[i]
    print('>>> Weight of Each Layer ', i+1, '\n')
    tmp = pd.DataFrame(array, columns=range(array.shape[1]))
    print (tmp, '\n\n')