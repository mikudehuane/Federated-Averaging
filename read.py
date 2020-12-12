from matplotlib import pyplot as plt
import pickle as pkl
import os.path as osp
import sys

file = sys.argv[1]
with open(file, 'rb') as f:
    img, label = pkl.load(f)
print(img.min(), img.max())
img = img.numpy().transpose(1,2,0) * 128 + 128
print(img.min(), img.max())
img = img.astype(int)
plt.imshow(img)
plt.savefig('.'.join([file, 'png']))
