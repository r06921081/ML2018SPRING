import numpy as np
import skimage
from skimage import io, transform
import sys
import os

size = 600
pics = []
image_dir = sys.argv[1]
recon_img = sys.argv[2]
imgNo = int(sys.argv[2].split('.')[0])

if len(sys.argv) >= 2:
    image_dir = sys.argv[1]
else:
    image_dir = './data/Aberdeen/'

filenames = os.listdir(image_dir)

filenames = list(map(lambda n: int(n.split('.')[0]),filenames))
filenames.sort()
for picName in filenames:
    pic = io.imread(os.path.join(image_dir, str(picName) + '.jpg'))
    pics.append(transform.resize(pic,(size,size), mode='constant'))

pic_num = len(pics)

flat_pics = np.reshape(pics, (pic_num, -1))
avgface = np.sum(flat_pics, axis=0)/pic_num
# plt.figure()
# plt.imshow(avgface.reshape(size,size,3))
# plt.show()

picMid = flat_pics - avgface # x - u
U, s, V = np.linalg.svd(picMid.T, full_matrices=False)
all_sigma = np.sum(s)

coordinate = np.dot(picMid, U) 

U_4 = U[:, :4]
avgface = avgface
coordinate = coordinate[:, :4]
targetImg = transform.resize(io.imread(os.path.join(image_dir, recon_img)), (size, size), mode='constant')


# plt.figure()
rcs = avgface + np.dot(coordinate[imgNo], U_4.T)
rcs -= np.min(rcs)
rcs /= np.max(rcs)
rcs = (rcs * 255).astype(np.uint8)
io.imsave('./reconstruction.png', rcs.reshape(size,size,3))

# plt.show()
