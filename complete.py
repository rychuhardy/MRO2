import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import random as rn
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from pylmnn.lmnn import LargeMarginNearestNeighbor as LMNN

repetitions_num = 50
euclidean1results = []
euclidean3results = []
mahanalobisresults = []
lmnnresults = []

data_points=[]
test_points=[]
n_neighbors = 1
points_per_class = 300
weights = 'uniform'

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for rep in range(0, repetitions_num):
    # Data generation
    mean = [5, 5]
    cov_matrix = np.matrix('15, 2; 5, 6')
    x, y = np.random.multivariate_normal(mean, cov_matrix, points_per_class).T
    data_points += list(zip(x, y, it.repeat(0)))

    mean = [-2, -2]
    cov_matrix = np.matrix('15, 1; 10, 6')
    x, y = np.random.multivariate_normal(mean, cov_matrix, points_per_class).T
    data_points += list(zip(x, y, it.repeat(1)))

    mean = [11, -3]
    cov_matrix = np.matrix('15, 5; 5, 5')
    x, y = np.random.multivariate_normal(mean, cov_matrix, points_per_class).T
    data_points += list(zip(x, y, it.repeat(2)))

    rn.shuffle(data_points)

    X = np.array(list(map(lambda point: [point[0], point[1]], data_points)))
    y = np.array(list(map(lambda point: point[2], data_points)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=19)
    covX = np.cov(X_train, rowvar=False)

    # Euclidean k=3

    clf = neighbors.KNeighborsClassifier(3, weights=weights)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
    euclidean3results.append(acc)

    #plt.show()

    # Euclidean k=1

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
    euclidean1results.append(acc)

    #plt.show()

    # Mahalanobis k = 1

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, metric='mahalanobis', metric_params={'V': covX})
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
    mahanalobisresults.append(acc)


    # lmnn
    clf = LMNN(n_neighbors=n_neighbors, max_iter=150, n_features_out=X.shape[1])
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(acc)
    lmnnresults.append(acc)


print("Euclidean k=1 std:", np.std(euclidean1results), " mean: ", np.mean(euclidean1results))
print("Euclidean k=3 std:", np.std(euclidean3results), " mean: ", np.mean(euclidean3results))
print("Mahanalobis k=1 std:", np.std(mahanalobisresults), " mean: ", np.mean(mahanalobisresults))
print("LMNN k=1 std:", np.std(lmnnresults), " mean: ", np.mean(lmnnresults))


# Plot results
N = 1
ind = np.arange(N)

width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(ind, np.mean(euclidean1results), width, color='red', yerr=np.std(euclidean1results))
rects2 = ax.bar(ind+width, np.mean(euclidean3results), width, color='blue', yerr=np.std(euclidean3results))
rects3 = ax.bar(ind+2*width, np.mean(mahanalobisresults), width, color='green', yerr=np.std(mahanalobisresults))
rects4 = ax.bar(ind+3*width, np.mean(lmnnresults), width, color='yellow', yerr=np.std(lmnnresults))

ax.set_ylabel('Accuracy of classifiers %')
ax.set_title('Classifiers accuracy')
ax.set_xticks(4*ind + width / 2)
ax.set_xticklabels('P1')

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('Euclidean k=1', 'Euclidean k=3', 'Mahanalobis k=1', 'LMNN k=1'))

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.show()