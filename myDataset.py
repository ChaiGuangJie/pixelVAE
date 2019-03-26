import torch, torchvision
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import myUtils
import h5py

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # img.show()
        greyImg = img.convert('L')  # L 灰色 RGB todo 转换到灰度以后，每个点的值代表的实际意义是？
        # greyImg.show()
        return greyImg


class ImageNet64Dataset(Dataset):
    def __init__(self, dir, transform, loader):
        images = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                images.append(path)
        self.images = images  # 为什么上面不直接 self.images = [] ?
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.images[index]
        image = self.loader(path)
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)

class ImageNet64DatasetH5(Dataset):
    def __init__(self, dir, transform, loader):
        self.f = h5py.File(dir,'r')
        self.images = self.f['data']
        self.transform = transform
        # self.loader = loader
        self.n_images = self.images.shape[0]//10

    def __getitem__(self, index):
        image = self.images[index].reshape((64,64,1))
        # image = self.loader(path)
        image = self.transform(image)
        return image

    def __len__(self):
        return self.n_images


class ImageNet64DatasetLoadAll(Dataset):
    def __init__(self, dir, transform, loader):
        images = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                image = loader(path)
                image = transform(image)
                images.append(image)

        self.images = images  # 为什么上面不直接 self.images = [] ?
        # self.transform = transform
        # self.loader = loader

    def __getitem__(self, index):
        # path = self.images[index]
        # image = self.loader(path)
        # image = self.transform(image)
        return self.images[index]

    def __len__(self):
        return len(self.images)



def CreateDataLoader(train_dir,test_dir,batch_size=128,num_workers=10):
    transform = Compose([ToTensor()])

    train_data = ImageNet64DatasetH5(train_dir, transform, pil_loader)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)  # imagenet64_dataLoader(train_data, 128)

    test_data = ImageNet64DatasetH5(test_dir, transform, pil_loader)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader,test_loader


if __name__ == "__main__":
    print(torch.__version__)

    train_dir = "/data1/home/guangjie/Data/imagenet64/train_64x64.hdf5"
    test_dir = "/data1/home/guangjie/Data/imagenet64/valid_64x64.hdf5"

    train_loader,test_loader = CreateDataLoader(train_dir,test_dir)

    for idx, data in enumerate(test_loader):
        print(idx)
        myUtils.save_image(data[:8].cpu(), 'tmp/testPic_' + str(idx) + '.png')
        if idx >= 1:
            break

    print('ok')
