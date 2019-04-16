import torch, torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, Pad
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import myUtils
import h5py
import numpy as np


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
    def __init__(self, dir, transform, loader=None, nc=1):
        self.f = h5py.File(dir, 'r')
        self.images = self.f['data']
        self.transform = transform
        self.nc = nc
        # self.loader = loader
        self.n_images = self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index].reshape((64, 64, self.nc))
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


def CreateDataLoader(train_dir, test_dir, batch_size=128, num_workers=4):
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    train_data = ImageNet64DatasetH5(train_dir, transform, pil_loader)
    train_loader = DataLoader(train_data, batch_size=batch_size,
                              num_workers=num_workers)  # imagenet64_dataLoader(train_data, 128)

    test_data = ImageNet64DatasetH5(test_dir, transform, pil_loader)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader


class Vim2DatesetAsGrey(Dataset):
    def __init__(self, path, transform):
        f = h5py.File(path, 'r')
        self.st = f['st']
        self.transform = transform
        self.n_imgs = self.st.shape[0]

    def __getitem__(self, item):
        # print(item)
        img = Image.fromarray(self.st[item].transpose((2, 1, 0))).convert('L').resize((64, 64))  # todo resize (64,64)
        return self.transform(img)

    def __len__(self):
        return self.n_imgs


def CreateVim2StDataloader(path, batch_size=128, num_workers=10):
    transform = Compose([ToTensor()])
    st_dataset = Vim2DatesetAsGrey(path, transform)
    return DataLoader(st_dataset, batch_size=batch_size, num_workers=num_workers)


class Vim2VaeHiddenDataset(Dataset):
    def __init__(self, path, time_step):
        f = h5py.File(path, 'r')
        self.mu_logvar = f['mu_logvar']  # shape=(108000, 8192)
        self.time_step = time_step
        self.n_sample = int(self.mu_logvar.shape[0] / self.time_step)

    def __getitem__(self, item):
        begin = item * self.time_step
        end = begin + self.time_step
        return torch.from_numpy(self.mu_logvar[begin:end, :])

    def __len__(self):
        return self.n_sample


def CreateVim2HiddenLoader(path, time_step=15, batch_size=128, num_workers=1):
    vim2hid_dateset = Vim2VaeHiddenDataset(path, time_step)
    return DataLoader(vim2hid_dateset, batch_size=batch_size, num_workers=num_workers)


class MovingMNISTDataset(Dataset):
    def __init__(self, path, transform):
        self.mv_mnist = np.load(path)  # shape = (20,10000,64,64)
        self.transform = transform
        self.data_shape = self.mv_mnist.shape
        self.shape_0 = self.data_shape[0]  # self.mv_mnist.shape[0]
        self.shape_1 = self.data_shape[1]  # self.mv_mnist.shape[1]

        self.n_imgs = self.shape_0 * self.shape_1

    def __getitem__(self, item):
        item_0 = item // self.shape_1
        item_1 = item % self.shape_0
        return self.transform(self.mv_mnist[item_0, item_1, :, :])

    def __len__(self):
        return self.n_imgs


def CreateMVMnistLoader(train_path, test_path, batch_size, num_workers=0):
    transform = Compose([ToTensor()])  # Normalize((0,), (1,))
    mv_mnist_train = MovingMNISTDataset(train_path, transform)
    train_loader = DataLoader(mv_mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    mv_mnist_test = MovingMNISTDataset(test_path, transform)
    test_loader = DataLoader(mv_mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader, test_loader


def CreateMVMnistAllLoader(path, batch_size=64, shuffle=False, num_workers=0):
    transform = Compose([ToTensor(), Normalize((0,), (1,))])
    mv_mnist_dataset = MovingMNISTDataset(path, transform)
    mv_mnist_loader = DataLoader(mv_mnist_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return mv_mnist_loader


class MvMnistHiddenDataset(Dataset):
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.hidden_z = self.f['z']

    def __getitem__(self, item):
        # seq_array = self.hidden_z[:, item, :]
        return torch.from_numpy(self.hidden_z[:, item, :])

    def __len__(self):
        return self.hidden_z.shape[1]


def CreateMvMnistHiddenDataset(path, batch_size, shuffle=True, num_workers=0):
    mv_mnist_hidden_dataset = MvMnistHiddenDataset(path)
    return DataLoader(mv_mnist_hidden_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)


class MvMnistHiddenDatasetSeq(Dataset):
    def __init__(self, path, timeStep=20):
        self.f = h5py.File(path, 'r')
        self.hidden_z = self.f['z']
        self.timeStep = timeStep
        self.hidLen = self.hidden_z.shape[1] // 20

    def __getitem__(self, item):
        beginItem = item * self.timeStep
        endItem = beginItem + self.timeStep
        # seq_array = self.hidden_z[:, item, :]
        return torch.from_numpy(self.hidden_z[:, beginItem:endItem, :])

    def __len__(self):
        return self.hidLen  # self.hidden_z.shape[1]//20


def CreateMvMnistHiddenSeqLoader(path, batch_size, shuffle=True, num_workers=0):
    mv_mnist_hidden_dataset = MvMnistHiddenDatasetSeq(path)
    return DataLoader(mv_mnist_hidden_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)


def CreateMnistDataloader(path, batch_size):
    transform = Compose([Pad(padding=2), ToTensor()])

    trainset = torchvision.datasets.MNIST(root=path, train=True,
                                          download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=path, train=False,
                                         download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


if __name__ == "__main__":
    print(torch.__version__)
    import cv2
    import scipy.misc

    # train_dir = "/data1/home/guangjie/Data/imagenet64/train_64x64.hdf5"
    # test_dir = "/data1/home/guangjie/Data/imagenet64/valid_64x64.hdf5"
    #
    # train_loader, test_loader = CreateDataLoader(train_dir, test_dir)
    #
    # for idx, data in enumerate(test_loader):
    #     print(idx)
    #     myUtils.save_image(data[:8].cpu(), 'tmp/testPic_' + str(idx) + '.png')
    #     if idx >= 1:
    #         break

    # img_path = "/data1/home/guangjie/Data/temp/logo.jpg"
    # img = cv2.imread(img_path)
    # # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("img_gray", img)

    # mv_mnist = np.load("/data1/home/guangjie/Data/mnist_test_seq.npy")
    # print(mv_mnist.shape)  # shape = (20,1000,64,64)


    # data_loader = CreateMVMnistAllLoader("/data1/home/guangjie/Data/MNIST/mv_mnist.hdf5",num_workers=4)
    # save_f = h5py.File("/data1/home/guangjie/Data/MNIST/mv_mnist_round.hdf5",'w')
    # data = save_f.create_dataset('data',shape=())
    # for step,data in enumerate(data_loader):


    # f = h5py.File("/data1/home/guangjie/Data/MNIST/mv_mnist.hdf5")  # "/data1/home/guangjie/Data/MNIST/mv_mnist_z.hdf5"
    # dataset = f.create_dataset('mv_mnist', data=mv_mnist)
    # f.close()

    # all_idx = [i for i in range(mv_mnist.shape[1])]
    # test_idx = all_idx[::10]
    # train_idx = list(set(all_idx) - set(test_idx))
    # train_mv_mnist = mv_mnist[:, train_idx, :, :]
    # test_mv_mnist = mv_mnist[:, test_idx, :, :]
    # print(train_mv_mnist.shape)
    # print(test_mv_mnist.shape)
    #
    # np.save("/data1/home/guangjie/Data/MNIST/mnist_train_seq.npy", train_mv_mnist)
    # np.save("/data1/home/guangjie/Data/MNIST/mnist_test_seq.npy", test_mv_mnist)

    # for i in range(3):
    #     # img = scipy.misc.toimage(mv_mnist[0, i, :, :])
    #     img_narray = mv_mnist[0, i, :, :]
    #     # cv2.imshow('mnist', np.reshape(img_narray, (64, 64, -1)))
    #     img = Image.fromarray(mv_mnist[0, i, :, :])
    #     img.show(img, 'test')
    #     print(i)
    print('ok')
