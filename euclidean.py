import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import random as rn
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.model_selection import train_test_split

print('dupa123')

data_points=[]
n_neighbors = 1

mean = [5, 5]
cov_matrix = np.matrix('10, 2; 5, 6')
x, y = np.random.multivariate_normal(mean, cov_matrix, 500).T
data_points += list(zip(x, y, it.repeat(0)))

mean = [-2, -2]
cov_matrix = np.matrix('10, 1; 10, 6')
x, y = np.random.multivariate_normal(mean, cov_matrix, 500).T
data_points += list(zip(x, y, it.repeat(1)))

mean = [11, -3]
cov_matrix = np.matrix('15, 5; 5, 5')
x, y = np.random.multivariate_normal(mean, cov_matrix, 500).T
data_points += list(zip(x, y, it.repeat(2)))

rn.shuffle(data_points)

X = np.array(list(map(lambda point: [point[0], point[1]], data_points)))
y = np.array(list(map(lambda point: point[2], data_points)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=19)

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    
    print(clf.score(X_test, y_test))

plt.show()