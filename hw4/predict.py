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
'''
argv[1] train data
argv[2] test data
argv[3] result dir
'''

X = np.load(sys.argv[1])

# X = X - np.mean(X)
X = X/255.
image = X
# x_val = X[:val_part].reshape((val_part, 28, 28, 1))
model = load_model('./new.h5')
model.summary()

encoder = Model(inputs=model.get_layer('input_1').input,
                                 outputs=model.get_layer('d3').output)

cnn_input = Input(model.get_layer('d4').input_shape[1:])
decoder = cnn_input
for layer in model.layers[9:]:
    decoder = layer(decoder)
decoder = Model(inputs=cnn_input, outputs=decoder)

encoded_imgs = encoder.predict(image[:10].reshape(-1,28,28,1))
image_X_autoencoder = encoder.predict(image.reshape(-1,28,28,1))
decoded_imgs = decoder.predict(encoded_imgs)


from sklearn.cluster import KMeans
import pandas as pd
# load testing data
test_case = pd.read_csv(sys.argv[2])
test_case.head()

# pca300 = PCA(n_components=100, whiten=True, svd_solver='full',random_state=0)
# pca300.fit(image_X_autoencoder)
# ppa300 = pca300.fit_transform(image_X_autoencoder) 


cluster_PCA = KMeans(n_clusters=2, n_jobs=12, random_state=0)
cluster_PCA.fit(image_X_autoencoder)
pt = cluster_PCA.fit_transform(image_X_autoencoder)

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