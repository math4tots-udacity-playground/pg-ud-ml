"""naive_bayes_example.py

As explained in the lessons, in classification problems facilitated
  by supervised learning with naive bayes, we have training data,
  and the machine learning algorithm provides us with a prediction
  surface.

'X' contains an array of points.
'Y' also an array, containing the same number of elements as 'X',
  describing what category the corresponding element in 'X' is
  categorized as.

Together, 'X' and 'Y' form a training set.

The specifics of how Gaussian naive bayes works is not yet clear
  to me, but from what I've gathered from the lesson so far,
  it is just one of many ways to generate a prediction surface.

"""
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))

X = np.array([
    [-1, -1, 4],
    [-2, -1, 4],
    [-3, -2, 5],
    [1, 1, 10],
    [2, 1, 11],
    [3, 2, 10]
])
Y = np.array(['boy', 'boy', 'boy', 'girl', 'girl', 'girl'])
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1, 10]]))
