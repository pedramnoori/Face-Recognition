import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

train_file = open('train.txt', 'r')
test_file = open('test.txt', 'r')
train_labels, train_data, test_labels, test_data = [], [], [], []

for line in train_file:
    image_file, label = line.strip().split()
    image = misc.imread(image_file)
    train_data.append(image.reshape(2500, ))
    train_labels.append(label)

train_data = np.array(train_data, dtype=float)
train_labels = np.array(train_labels, dtype=int)

for line in test_file:
    image_file, label = line.strip().split()
    image = misc.imread(image_file)
    test_data.append(image.reshape(2500, ))
    test_labels.append(label)

test_data = np.array(test_data, dtype=float)
test_labels = np.array(test_labels, dtype=int)

plt.imshow(train_data[5, :].reshape(50 , 50), cmap=cm.Greys_r)
plt.show()

plt.imshow(test_data[5, :].reshape(50 , 50), cmap=cm.Greys_r)
plt.show()

train_data_sum = np.zeros((1, train_data.shape[1]), dtype=float)
for i in range(train_data.shape[0]):
    train_data_sum += train_data[i]
mean_vals = train_data_sum / train_data.shape[0]

plt.imshow(mean_vals.reshape(50 , 50), cmap=cm.Greys_r)
plt.show()

mean_removed_train_data = train_data - mean_vals
mean_removed_test_data = test_data - mean_vals

plt.imshow(mean_removed_train_data[5, :].reshape(50 , 50), cmap=cm.Greys_r)
plt.show()

plt.imshow(mean_removed_test_data[5, :].reshape(50 , 50), cmap=cm.Greys_r)
plt.show()

U, s, V = np.linalg.svd(mean_removed_train_data)

def to_diagonal(mat):
    size = int(len(mat))
    ret = np.zeros((size, size))
    for i in range(size):
        ret[i, i] = mat[i]
    return ret

s = to_diagonal(s)

for i in range(10):
    plt.imshow(V[i, :].reshape(50, 50), cmap=cm.Greys_r)
    plt.show()

lowrank_error = []
for r in range(1, 201):
    X = np.dot(np.dot(U[:, :r], s[:r, :r]), V[:r, :])
    dist = np.linalg.norm(X - mean_removed_train_data)
    lowrank_error.append(dist)

plt.plot(lowrank_error)
plt.ylabel('rank-r approximation error')
plt.xlabel('R')
plt.show()

def generateF(V, X, r):
    VT = V[:r, :].T
    return np.dot(X, VT)

reg = OneVsRestClassifier(LogisticRegression()).fit(generateF(V, train_data, 10), train_labels)
print(reg.score(generateF(V, test_data, 10), test_labels))

accuracy = []

for r in range(1, 10):
    OVR = OneVsRestClassifier(LogisticRegression()).fit(generateF(V, train_data, r), train_labels)
    accuracy.append(OVR.score(generateF(V, test_data, r), test_labels))

plt.plot(accuracy)
plt.show()