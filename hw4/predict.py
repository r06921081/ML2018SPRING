import numpy as np
import sys
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
# import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn import cluster
from sklearn.manifold import Isomap
from tools import p
import dataprocess as dp
from sklearn.decomposition import PCA


X = np.load(sys.argv[1])

# X = X - np.mean(X)
X = X/255.
image = X
# x_val = X[:val_part].reshape((val_part, 28, 28, 1))
model = load_model('./new.h5')
model.summary()

encoder = Model(inputs=model.get_layer('input_1').input,
                                 outputs=model.get_layer('d3').output)

DL_input = Input(model.get_layer('d4').input_shape[1:])
decoder = DL_input
for layer in model.layers[9:]:
    decoder = layer(decoder)
decoder = Model(inputs=DL_input, outputs=decoder)

encoded_imgs = encoder.predict(image[:10].reshape(-1,28,28,1))
image_X_autoencoder = encoder.predict(image.reshape(-1,28,28,1))
decoded_imgs = decoder.predict(encoded_imgs)


# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(image[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)


# from sklearn.decomposition import PCA
# pca = PCA(n_components=32, whiten=True,svd_solver="full")
# image_X_PCA = pca.fit_transform(image)
# decoded_imgs = pca.inverse_transform(image_X_PCA[:10])
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(image[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

from sklearn.cluster import KMeans
import pandas as pd
# load testing data
test_case = pd.read_csv(sys.argv[2])
test_case.head()

pca300 = PCA(n_components=100, whiten=True, svd_solver='full',random_state=0)
pca300.fit(image_X_autoencoder)
ppa300 = pca300.fit_transform(image_X_autoencoder) 


cluster_PCA= KMeans(n_clusters=2)
cluster_PCA.fit(ppa300)
pt = cluster_PCA.fit_transform(ppa300)

def clust_ans(cluster):
    ans = []
    for a, b in zip(test_case["image1_index"],test_case["image2_index"]):
        if cluster.labels_[a] == cluster.labels_[b]:
            ans.append(1)
        else:
            ans.append(0)
    return ans
# submission
dp.savepre(clust_ans(cluster_PCA), sys.argv[3])
p('9')