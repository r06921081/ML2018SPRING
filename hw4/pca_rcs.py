import numpy as np
import skimage
from skimage import io, transform
# import matplotlib.pyplot as plt
import sys
import os

size = 600
image_dir = sys.argv[1]
recon_img = sys.argv[2]
imgNo = int(sys.argv[2].split('.')[0])

U_4 = np.load('./U_4.npy')
avgface = np.load('./avgface.npy')
coordinate = np.load('./coor.npy')
targetImg = transform.resize(io.imread(os.path.join(image_dir, recon_img)), (size, size), mode='constant')


# plt.figure()
rcs = avgface + np.dot(coordinate[imgNo], U_4.T)
rcs -= np.min(rcs)
rcs /= np.max(rcs)
rcs = (rcs * 255).astype(np.uint8)
io.imsave('reconstruction.jpg', rcs.reshape(size,size,3))

# plt.imshow(rcs.reshape(size,size,3))
# plt.title("pic:" + str(imgNo))
# plt.xticks([])
# plt.yticks([])
# plt.tight_layout()

# plt.show()

