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
import visdom

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from myDataset import CreateDataLoader, CreateMVMnistLoader, CreateMnistDataloader
import models, util
from vae_model import LinearVAE

vis = visdom.Visdom(server='http://172.18.29.70', env='cnn-vae')
assert vis.check_connection()


class VAE(nn.Module):
    def __init__(self, H, W, zsize, vae_depth, colors, out_ch=1):
        super(VAE, self).__init__()
        self.vae_encoder = models.ImEncoder(in_size=(H, W), zsize=zsize, use_bn=True, depth=vae_depth,
                                            colors=colors)
        # todo self.vae_encoder 输出的隐变量是拼接在一起的！
        self.vae_decoder = models.ImDecoder(in_size=(H, W), zsize=zsize, use_bn=True, depth=vae_depth,
                                            out_channels=out_ch)
        # self.pixcnn = models.LGated((C, H, W), OUTCN, 0, num_layers=arg.num_layers, k=krn, padding=pad)
        # self.eblock = util.Block(1, 16, 3)
        # self.flt = util.Flatten()
        # self.fc_mu = nn.Linear(4096 * 16, args.zsize)
        # self.fc_logvar = nn.Linear(4096 * 16, args.zsize)

        # self.efc1 = nn.Linear(4096, 8192)
        # self.efc2 = nn.Linear(8192, 8192)
        # self.efc31 = nn.Linear(8192, args.zsize)
        # self.efc32 = nn.Linear(8192, args.zsize)
        # self.dfc = nn.Linear(args.zsize, 4096 * 16)
        # self.rsp = util.Reshape((16, 64, 64))
        # self.dblock = util.Block(16, 32, deconv=True)
        # # self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.convTrsp = nn.ConvTranspose2d(32, 1, 1)

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

    # def vae_decoder(self, z):
    #     o = F.relu(self.dfc(z))
    #     o = self.rsp(o)
    #     o = self.dblock(o)
    #     return F.sigmoid(self.convTrsp(o))
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
        mu, logvar = self.vae_encoder(x.view(-1, C, H, W))
        z = self.reparameterize(mu, logvar)
        # return self.vae_decoder(z), mu, logvar
        return self.vae_decoder(z), mu, logvar
        # mu, logvar = self.encode(x.view(-1, 4096))
        # z = self.reparameterize(mu, logvar)
        # return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, C, H, W), x, reduction='sum')  # x.view(-1, 1, 64, 64)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def BCELoss(recon_x, x):
    return F.binary_cross_entropy(recon_x.view(-1, C, H, W), x, reduction='sum')  # x.view(-1, 1, 64, 64)


def KLDLoss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

MSELoss = nn.MSELoss()

train_origin_rec_win = vis.images(torch.zeros(16, 1, 64, 64, dtype=torch.float).cpu(),
                                  opts={"title": "origin rec images"})
train_bce_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), opts={"title": "train_bce_loss"})
train_kld_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), opts={"title": "train_kld_loss"})
train_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), opts={"title": "train_loss"})


def train(epoch, train_loader):
    model.train()
    global global_idx, train_origin_rec_win, train_bce_loss_win, train_kld_loss_win, train_loss_win
    # tbw_idx = epoch * iterPerEpoch
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        # data = data.view(-1, 1, 64, 64)
        data = data[0].to(device) # data.to(device)  #
        # data = data.view(-1, H * W)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # loss = loss_function(recon_batch, data, mu, logvar)
        bce_loss = MSELoss(recon_batch,data) #BCELoss(recon_batch, data)  # todo 因子对训练结果的影响至关重要?
        kld_loss = KLDLoss(mu, logvar)  # todo 如这里改为 *100 训练没有效果?
        loss = bce_loss + kld_loss
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

            n = min(data.shape[0], 8)
            vis.images(torch.cat((data[:n], recon_batch[:n])).cpu() * 0.5 + 0.5, win=train_origin_rec_win)
            vis.line(Y=torch.tensor([train_loss]), X=global_idx, win=train_loss_win, update="append")
            vis.line(Y=torch.tensor([bce_loss]), X=global_idx, win=train_bce_loss_win, update="append")
            vis.line(Y=torch.tensor([kld_loss]), X=global_idx, win=train_kld_loss_win, update="append")
        # tbw.add_scalar('cnn-vae/train/kld_loss', kld_loss.item() / len(data),
        #                global_idx)
        # tbw.add_scalar('cnn-vae/train/bce_loss', bce_loss.item() / len(data),
        #                global_idx)

        global_idx += 1

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


test_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="test_loss", opts={"title": "test_loss"})
test_origin_rec_win = vis.images(torch.zeros(16, 1, 64, 64, dtype=torch.float).cpu(),
                                 opts={"title": "origin rec images"})


def test(epoch, test_loader):
    global test_loss_win, test_origin_rec_win
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # data = data.view(-1, 1, 64, 64)
            data = data[0].to(device) #data.to(device)  #
            # data = data.view(-1, H * W)
            recon_batch, mu, logvar = model(data)
            bce_loss = BCELoss(recon_batch, data)  # todo 因子对训练结果的影响至关重要?
            kld_loss = KLDLoss(mu, logvar)  # todo 如这里改为 *100 训练没有效果?
            loss = bce_loss + kld_loss
            test_loss += loss / len(data)
            # test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i % args.log_interval == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data.view(args.batch_size, 1, H, W)[-n:],
                                        recon_batch.view(args.batch_size, 1, H, W)[-n:]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)
                vis.line(Y=torch.tensor([test_loss]), X=torch.tensor([epoch]), win=test_loss_win, update="append")

                vis.images(torch.cat((data[:n], recon_batch[:n])).cpu() * 0.5 + 0.5, win=test_origin_rec_win)

    test_loss /= len(test_loader.dataset)
    vis.line(Y=torch.tensor([test_loss]), X=global_idx)
    # tbw.add_scalar('cnn-vae/test/test_loss', test_loss, epoch)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
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
                        default=32, type=int)

    parser.add_argument("-d", "--vae-depth",
                        dest="vae_depth",
                        help="Depth of the VAE in blocks (in addition to the 3 default blocks)",
                        default=0, type=int)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # train_path = "/data1/home/guangjie/Data/imagenet64/train_64x64.hdf5"
    # test_path = "/data1/home/guangjie/Data/imagenet64/valid_64x64.hdf5"
    # train_loader, test_loader = CreateDataLoader(train_path, test_path)

    # mv_mnist_train_path = "/data1/home/guangjie/Data/MNIST/mnist_train_seq.npy"
    # mv_mnist_test_path = "/data1/home/guangjie/Data/MNIST/mnist_test_seq.npy"
    # train_loader, test_loader = CreateMVMnistLoader(mv_mnist_train_path, mv_mnist_test_path, batch_size=args.batch_size,
    #                                                 num_workers=6)

    train_loader, test_loader = CreateMnistDataloader("/data1/home/guangjie/Project/python/pixel-models/data/MNIST",
                                                      batch_size=args.batch_size)

    C, H, W =  1, 32, 32  #1, 64, 64  #
    OUTCN = 16  # 64
    iterPerEpoch = len(train_loader)
    print(iterPerEpoch)
    # log_dir = './runs/vae_mv_mnist' + str(int(time.time()))
    # tbw = SummaryWriter(log_dir=log_dir)

    global_idx = torch.tensor([0])

    # model = LinearVAE(H, W, args.zsize).to(device)

    model = VAE(H, W, args.zsize, args.vae_depth, C).to(device)


    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         # torch.nn.init.xavier_uniform(m.weight)
    #         nn.init.uniform_(m.weight)
    #         # m.weight.data.fill_(0.01)
    #         m.bias.data.fill_(0.01)
    #     if isinstance(m, nn.Conv2d):
    #         torch.nn.init.xavier_uniform(m.weight)
    #         # torch.nn.init.xavier_uniform(m.bias)
    def weight_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # if m.bias:
            #     m.bias.data.fill_(0.01)
            #     torch.nn.init.xavier_uniform_(m.bias)


    model.apply(weight_init)  #

    # load_model_path = "/data1/home/guangjie/Project/python/pixel-models/savedModels/cnn-vae-mv-imagenet64-1555041800-1024-e5.pt"
    # model.load_state_dict(torch.load(load_model_path))
    # print('load model --- ' + load_model_path)

    # model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(model)
    n_sample = 32
    sample_win = vis.images(torch.zeros(n_sample, 1, 64, 64, dtype=torch.float).cpu(), opts={"title": "gen images"})
    save_path = 'savedModels/cnn-vae-mv-imagenet64-' + str(int(time.time())) + '.pt'

    for epoch in range(1, args.epochs + 1):
        train(epoch, train_loader)
        test(epoch, test_loader)
        with torch.no_grad():
            sample = torch.randn(n_sample, args.zsize).to(device)
            sample = model.vae_decoder(sample).cpu()  # .forward vae_
            save_image(sample.view(32, 1, H, W),
                       'results/sample_' + str(epoch) + '.png')
            sample_win = vis.images(sample[:n_sample].cpu() * 0.5 + 0.5, win=sample_win)

        # torch.save(model.state_dict(), save_path)
        # tbw.close()
        # print('model saved --- ' + save_path)
