from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
import torchvision
from torchvision.transforms import ToTensor, Compose
from torchvision.utils import save_image
import os, time
from myDataset import CreateDataLoader, MovingMNISTDataset
import torchvision.transforms as transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument("-D", "--data-directory",
                    dest="data_dir",
                    help="Data directory",
                    default='/data1/home/guangjie/Data/imagenet64/', type=str)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

'''
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
'''
# train_path = "/data1/home/guangjie/Data/imagenet64/train_64x64.hdf5"
# test_path = "/data1/home/guangjie/Data/imagenet64/valid_64x64.hdf5"
#
# train_loader, test_loader = CreateDataLoader(train_path, test_path)
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
dataset = MovingMNISTDataset("/data1/home/guangjie/Data/MNIST/mnist_train_seq.npy", transform)
test_dataset = MovingMNISTDataset("/data1/home/guangjie/Data/MNIST/mnist_test_seq.npy", transform)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                           shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                          shuffle=True, num_workers=4)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc31 = nn.Linear(128, 64)
        self.fc32 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 512)
        self.fc6 = nn.Linear(512, 4096)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 4096))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        nn.init.uniform_(m.weight, -0.01, 0.01)
        nn.init.uniform_(m.bias, -0.01, 0.01)


model = VAE().to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
MSECriterion = nn.MSELoss(reduction='sum')


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 4096), reduction='sum')
    MSE = MSECriterion(recon_x, x.view(-1, 4096))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return KLD + MSE


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
        if batch_idx == 2812:
            n = min(data.size(0), 8)
            # recon_batch = recon_batch.mul_(172.5).clamp_(0, 255)
            comparison = torch.cat([data[:n],
                                    recon_batch.view(data.shape[0], 1, 64, 64)[:n]])
            save_image(comparison.cpu(),
                       'myResults/train_reconstruction_' + str(epoch) + '.png', nrow=n)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                # recon_batch = recon_batch.mul_(172.5).clamp_(0, 255)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(data.shape[0], 1, 64, 64)[:n]])
                save_image(comparison.cpu(),
                           'myResults/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 64).to(device)
            sample = model.decode(sample).cpu()
            # sample = sample.mul_(172.5).clamp_(0, 255)
            save_image(sample.view(64, 1, 64, 64),
                       'myResults/sample_' + str(epoch) + '.png')
    torch.save(model.state_dict(), 'savedModels/orgin-vae-mvmnist-' + str(int(time.time())) + '.pt')
    print('model saved!')
