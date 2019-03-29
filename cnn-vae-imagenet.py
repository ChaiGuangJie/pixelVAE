from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
# from torchvision import datasets, transforms
import torchvision
from torchvision.transforms import ToTensor, Compose
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import os, time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from myDataset import CreateDataLoader
import models, util

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument("-D", "--data-directory",
                    dest="data_dir",
                    help="Data directory",
                    default='/data1/home/guangjie/Data/imagenet64/', type=str)

parser.add_argument("-T", "--tb-directory",
                    dest="tb_dir",
                    help="Tensorboard directory",
                    default='./runs/vae', type=str)

parser.add_argument("-z", "--z-size",
                    dest="zsize",
                    help="Size of latent space.",
                    default=4096, type=int)

parser.add_argument("-d", "--vae-depth",
                    dest="vae_depth",
                    help="Depth of the VAE in blocks (in addition to the 3 default blocks)",
                    default=0, type=int)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

train_path = "/data1/home/guangjie/Data/imagenet64/train_64x64.hdf5"
test_path = "/data1/home/guangjie/Data/imagenet64/valid_64x64.hdf5"

train_loader, test_loader = CreateDataLoader(train_path, test_path)
C, H, W = 1, 64, 64
OUTCN = 64
iterPerEpoch = len(train_loader)
print(iterPerEpoch)
log_dir = './runs/vae_' + str(int(time.time()))
tbw = SummaryWriter(log_dir=log_dir)
global_idx = 0


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.vae_encoder = models.ImEncoder(in_size=(H, W), zsize=args.zsize, use_bn=False, depth=args.vae_depth,
                                            colors=C)
        # todo self.vae_encoder 输出的隐变量是拼接在一起的！
        self.decoder = models.ImDecoder(in_size=(H, W), zsize=args.zsize, use_bn=False, depth=args.vae_depth,
                                        out_channels=1)
        # self.pixcnn = models.LGated((C, H, W), OUTCN, 0, num_layers=arg.num_layers, k=krn, padding=pad)
        # self.eblock = util.Block(1, 16, 3)
        # self.flt = util.Flatten()
        # self.fc_mu = nn.Linear(4096 * 16, args.zsize)
        # self.fc_logvar = nn.Linear(4096 * 16, args.zsize)

        # self.efc1 = nn.Linear(4096, 8192)
        # self.efc2 = nn.Linear(8192, 8192)
        # self.efc31 = nn.Linear(8192, args.zsize)
        # self.efc32 = nn.Linear(8192, args.zsize)
        self.dfc = nn.Linear(args.zsize, 4096 * 16)
        self.rsp = util.Reshape((16, 64, 64))
        self.dblock = util.Block(16, 32, deconv=True)
        # self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear')
        self.convTrsp = nn.ConvTranspose2d(32, 1, 1)

        # self.dfc1 = nn.Linear(args.zsize, 4096)
        # self.dfc2 = nn.Linear(4096, 8192)
        # self.dfc3 = nn.Linear(8192, 4096)

    # def vae_encoder(self, x):
    #     e = self.eblock(x.view(-1, 1, 64, 64))  # 无pooling
    #     e = self.flt(e)
    #     mu = self.fc_mu(e)
    #     logvar = self.fc_logvar(e)
    #     return mu, logvar

    # o = F.relu(self.efc1(x))
    # o = F.relu(self.efc2(o))
    # return self.efc31(o), self.efc32(o)

    def vae_decoder(self, z):
        o = F.relu(self.dfc(z))
        o = self.rsp(o)
        o = self.dblock(o)
        return F.sigmoid(self.convTrsp(o))
        # o = F.relu(self.dfc1(z))
        # o = F.relu(self.dfc2(o))
        # return F.sigmoid(self.dfc3(o))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # e = self.eblock(x.view(-1, 1, 64, 64))  # 无pooling
        # e = self.flt(e)
        # mu = self.fc_mu(e)
        # logvar = self.fc_logvar(e)
        # mu, logvar = self.vae_encoder(x.view(-1, 4096)) #.forward view(-1, 1, 64, 64)
        # mu, logvar = self.encoder(x)  # .view(-1,1,64,64)
        # z = util.sample(mu, logvar)
        mu, logvar = self.vae_encoder(x.view(-1, 1, 64, 64))
        z = self.reparameterize(mu, logvar)
        # return self.vae_decoder(z), mu, logvar
        return self.decoder(z), mu, logvar
        # mu, logvar = self.encode(x.view(-1, 4096))
        # z = self.reparameterize(mu, logvar)
        # return self.decode(z), mu, logvar


model = VAE().to(device)
load_model_path = "/data1/home/guangjie/Project/python/pixel-models/savedModels/cnn-vae-imagenet-1553764979-epoch50.pt"
model.load_state_dict(torch.load(load_model_path))
print('load model --- '+load_model_path)
# model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 1, 64, 64), x, reduction='sum')  # x.view(-1, 1, 64, 64)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def BCELoss(recon_x, x):
    return F.binary_cross_entropy(recon_x.view(-1, 1, 64, 64), x, reduction='sum')  # x.view(-1, 1, 64, 64)


def KLDLoss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def train(epoch):
    model.train()
    global global_idx
    # tbw_idx = epoch * iterPerEpoch
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        # data = data.view(-1, 1, 64, 64)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # loss = loss_function(recon_batch, data, mu, logvar)
        bce_loss = BCELoss(recon_batch, data)
        kld_loss = KLDLoss(mu, logvar)
        loss = bce_loss + kld_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))
        tbw.add_scalar('cnn-vae-imagenet/train/kld_loss', kld_loss.item() / len(data),
                       global_idx)
        tbw.add_scalar('cnn-vae-imagenet/train/bce_loss', bce_loss.item() / len(data),
                       global_idx)

        global_idx += 1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # data = data.view(-1, 1, 64, 64)
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 64, 64)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    tbw.add_scalar('cnn-vae-imagenet/test/test_loss', test_loss, epoch)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    print(model)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, args.zsize).to(device)
            sample = model.decoder(sample).cpu()  # .forward vae_
            save_image(sample.view(64, 1, 64, 64),
                       'results/sample_' + str(epoch) + '.png')
    save_path = 'savedModels/cnn-vae-imagenet-' + str(int(time.time())) + '.pt'
    torch.save(model.state_dict(), save_path)
    tbw.close()
    print('model saved --- ' + save_path)
