import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import Birch, KMeans
from sklearn import cluster
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tools import p
import dataprocess as dp
from matplotlib import offsetbox

test_x = dp.readtest("./data/test_case.csv")

X = np.load('./data/image.npy')
X = X.reshape(X.shape[0], -1)/255.
# X = X[:20000]
# test = dp.readtest(sys.argv[1])
# print(test.shape)

print(X.shape[0])
last = 0
pca = PCA(n_components=410, whiten=True, svd_solver='full',random_state=0)

ppa = pca.fit_transform(X) 

decoded_imgs = pca.inverse_transform(ppa[:10])
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.show()

# clustering 
cluster_PCA= KMeans(n_clusters=2)
cluster_PCA.fit(ppa)
pt = cluster_PCA.fit_transform(ppa)

result = []
for row in test_x:
    if cluster_PCA.labels_[row[0]] == cluster_PCA.labels_[row[1]]:
        result.append(1)
    else:
        result.append(0)

dp.savepre(result, './result/test.csv')
plt.figure()
plt.scatter(pt[:, 0], pt[:, 1], c = cluster_PCA.labels_ )
plt.show()