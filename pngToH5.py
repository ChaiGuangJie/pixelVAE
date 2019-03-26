import os
from PIL import Image
from torchvision.transforms import ToTensor, Compose
from scipy import io as sio
import numpy as np


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # img.show()
        greyImg = img.convert('L')  # L 灰色 RGB todo 转换到灰度以后，每个点的值代表的实际意义是？
        # greyImg.show()
        return greyImg


# # transform = Compose([ToTensor()])
train_dir = "/data1/home/guangjie/Data/imagenet64/train_64x64/"
valid_dir = "/data1/home/guangjie/Data/imagenet64/valid_64x64/"
# images = []
# for root, _, fnames in sorted(os.walk(test_dir)):
#     for fname in sorted(fnames):
#         path = os.path.join(root, fname)
#         image = pil_loader(path)
#         # image = transform(image)
#         image_array = np.array(image)
#         images.append(image_array)
#
# # sio.savemat("/data1/home/guangjie/Data/imagenet64/train_64x64.mat",{'data':images}) #too larger
# print('saved!')

import h5py

# import numpy as np

f = h5py.File("/data1/home/guangjie/Data/imagenet64/valid_64x64.hdf5", "w")
dset = f.create_dataset('data', (49999, 4096), dtype=np.uint8)
for root, _, fnames in sorted(os.walk(valid_dir)):
    for idx, fname in enumerate(sorted(fnames)):
        path = os.path.join(root, fname)
        image = pil_loader(path)
        # image = transform(image)
        image_array = np.array(image)
        # images.append(image_array)
        dset[idx, :] = image_array.reshape(-1)
        if idx % 5000 == 0:
            print(idx)
f.close()

print('saved!')
