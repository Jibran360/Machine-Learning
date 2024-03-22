from turtle import title
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
iris = datasets.load_iris()

x = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict(([[2.6]]))
print(example)

# Using Matplotlib to plot the visualization
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)
plt.plot(X_new, y_prob[:,1], "g-")
plt.title("virginica")
plt.show()

#print(y)


#print(iris["data"])
#print(x)