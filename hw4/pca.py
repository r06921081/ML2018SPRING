import numpy as np
import skimage
from skimage import io, transform
# import matplotlib.pyplot as plt
import sys
import os

size = 600
pics = []
image_dir = sys.argv[1]
recon_img = sys.argv[2]
imgNo = int(sys.argv[2].split('.')[0])

if len(sys.argv) >= 2:
    image_dir = sys.argv[1]
    if image_dir[len(image_dir)-1] != '/':
        image_dir += '/'
else:
    image_dir = './data/Aberdeen/'

for i in range(0, 415):
    pic = io.imread(os.path.join(image_dir, str(i) + '.jpg'))
    pics.append(transform.resize(pic,(size,size), mode='constant'))
pic_num = len(pics)

flat_pics = np.reshape(pics, (pic_num, -1))
avgface = np.sum(flat_pics, axis=0)/pic_num
# plt.figure()
# plt.imshow(avgface.reshape(size,size,3))
# plt.show()

pic_center = flat_pics - avgface # x - u
U, s, V = np.linalg.svd(pic_center.T, full_matrices=False)
all_sigma = np.sum(s)

coordinate = np.dot(pic_center, U) 

# fig = plt.figure()
# for i, imgNo in enumerate([0,100,200,300]):
#     imgs = pics[imgNo]
#     # total = fig.add_subplot(2, 2, i + 1)
#     imgs -= np.min(imgs)
#     imgs /= np.max(imgs)
#     imgs = (imgs * 255).astype(np.uint8)
#     # total.imshow(imgs.reshape(size,size,3))
#     # plt.title("pic:"+str(imgNo))
#     # plt.xticks([])
#     # plt.yticks([])
#     # plt.tight_layout()

# # fig = plt.figure()
# for i, imgNo in enumerate([0,100,200,300]):
#     rcs = avgface + np.dot(coordinate[imgNo, :4], U[:, :4].T)
#     # total = fig.add_subplot(2, 2, i + 1)
#     rcs -= np.min(rcs)
#     rcs /= np.max(rcs)
#     rcs = (rcs * 255).astype(np.uint8)
# #     total.imshow(rcs.reshape(size,size,3))
# #     plt.title("pic:"+str(imgNo))
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.tight_layout()
# # plt.show()

# for i, u in enumerate(U.T):
#     print(i + 1, s[i]/all_sigma)
#     l = u.reshape(size, size, 3).copy()
#     l -= np.min(l)
#     l /= np.max(l)
#     l = (l * 255).astype(np.uint8)
#     # plt.figure()
#     # plt.imshow(l)
#     # plt.figure()
#     # plt.imshow(255-l)
#     if i == 3:
#         break


U_4 = U[:, :4]
avgface = avgface
coordinate = coordinate[:, :4]
targetImg = transform.resize(io.imread(os.path.join(image_dir, recon_img)), (size, size), mode='constant')


# plt.figure()
rcs = avgface + np.dot(coordinate[imgNo], U_4.T)
rcs -= np.min(rcs)
rcs /= np.max(rcs)
rcs = (rcs * 255).astype(np.uint8)
io.imsave('reconstruction.png', rcs.reshape(size,size,3))

# plt.show()
