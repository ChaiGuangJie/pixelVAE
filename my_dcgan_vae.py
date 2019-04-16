from __future__ import print_function
import argparse
import os
import time
import random
import math
import torch
import torch.nn as nn
# import torch.legacy.nn as lnn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import visdom, cv2
from torch.autograd import Variable
from myDataset import MovingMNISTDataset, ImageNet64DatasetH5
# from viewImages import showImg
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torchvision.utils import save_image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vis = visdom.Visdom(server='http://172.18.29.70', env='vae_dcgan')
assert vis.check_connection()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)  # todo  64
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=24, help='number of epochs to train for')  # todo
parser.add_argument('--saveInt', type=int, default=5, help='number of epochs between checkpoints')
parser.add_argument('--lr', type=float, default=0.00006, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nlog', type=int, default=30, help="number of update log")

# --netD
# "/data1/home/guangjie/Project/python/pixel-models/outf/netD_epoch_8_1555153455.pth"
# --netG
# "/data1/home/guangjie/Project/python/pixel-models/outf/netG_epoch_8_1555153455.pth"

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# log_dir = './runs/dcgan_vae_mv_mnist' + str(int(time.time()))
# tbw = SummaryWriter(log_dir=log_dir)

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                           )
elif opt.dataset == 'mnist32':
    transform = transforms.Compose([transforms.Pad(padding=2), transforms.ToTensor()])
    dataset = torchvision.datasets.MNIST(root="/data1/home/guangjie/Project/python/pixel-models/data/MNIST", train=True,
                                         download=False, transform=transform)

elif opt.dataset == 'mvmnist':
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MovingMNISTDataset(opt.dataroot, transform)

elif opt.dataset == "imagenet64":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = ImageNet64DatasetH5(opt.dataroot, transform)
    #

else:
    dataset = None

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1  # 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def KLD_loss(input):
    mu = input[0]
    logvar = input[1]
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return KLD


def KLDLoss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()

    def forward(self, mu_logvar):
        mu = mu_logvar[0]
        logvar = mu_logvar[1]

        std = logvar.mul(0.5).exp_()  # calculate the STDEV
        if opt.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class _Encoder(nn.Module):
    def __init__(self, imageSize):
        super(_Encoder, self).__init__()

        n = math.log2(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.conv1 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2 ** (n - 3), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv', nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))
        for i in range(n - 3):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)),
                                    nn.Conv2d(ngf * 2 ** (i), ngf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i + 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i + 1)))
            self.encoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*8) x 4 x 4

    def forward(self, input):
        output = self.encoder(input)
        return [self.conv1(output), self.conv2(output)]


class _netG(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(imageSize)
        self.sampler = _Sampler()

        n = math.log2(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)

        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2 ** (n - 3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2 ** (n - 3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n - 3, 0, -1):
            self.decoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i - 1)),
                                    nn.ConvTranspose2d(ngf * 2 ** i, ngf * 2 ** (i - 1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i - 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i - 1)))
            self.decoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i - 1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            e = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.sampler, e, range(self.ngpu))
            output = nn.parallel.data_parallel(self.decoder, output, range(self.ngpu))
        else:
            e = self.encoder(input)
            output = self.sampler(e)
            output = self.decoder(output)
        return output, e

    def make_cuda(self):
        self.encoder.cuda()
        self.sampler.cuda()
        self.decoder.cuda()


netG = _netG(opt.imageSize, ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
    print("load netG")

print(netG)


class _netD(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        n = math.log2(imageSize)

        assert n == round(n), 'imageSize must be a power of 2'
        assert n >= 3, 'imageSize must be at least 8'
        n = int(n)
        self.main = nn.Sequential()

        # input is (nc) x 64 x 64
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf) x 32 x 32
        for i in range(n - 3):
            self.main.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** (i), ngf * 2 ** (i + 1)),
                                 nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i + 1), 4, 2, 1, bias=False))
            self.main.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i + 1)), nn.BatchNorm2d(ndf * 2 ** (i + 1)))
            self.main.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i + 1)), nn.LeakyReLU(0.2, inplace=True))

        self.main.add_module('output-conv', nn.Conv2d(ndf * 2 ** (n - 3), 1, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)


netD = _netD(opt.imageSize, ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
    print("load netD")
print(netD)

criterion = nn.BCELoss(weight=torch.full((opt.batchSize, 1), 10), reduction="sum")  #
MSECriterion = nn.MSELoss(reduction="sum")  # reduction="sum"   # todo sum是关键！！！！！，不然重建的mnist图像训练不出来
# BCECriterion = nn.BCELoss()
# def BCELoss(recon_x, x):
#     return F.binary_cross_entropy(recon_x, x)

# input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
# noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
# fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
# label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.make_cuda()
    criterion.cuda()
    MSECriterion.cuda()
    # input, label = input.cuda(), label.cuda()
    # noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# input = Variable(input)
# label = Variable(label)
# noise = Variable(noise)
# fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# origin_win = vis.images(torch.zeros(8, 1, 64, 64, dtype=torch.float).cpu(), opts={"title": "origin images"})
# rec_win = vis.images(torch.zeros(8, 1, 64, 64, dtype=torch.float).cpu(), opts={"title": "rec images"})
origin_rec_win = vis.images(torch.zeros(16, 1, 64, 64, dtype=torch.float).cpu(), opts={"title": "origin rec images"})
gen_win = vis.images(torch.zeros(8, 1, 64, 64, dtype=torch.float).cpu(), opts={"title": "gen images"})

VAE_MSE_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="VAEerr_win", opts={"title": "VAE_MSE_win"})
VAE_KLD_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="VAEerr_win", opts={"title": "VAE_KLD_win"})

errD_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="errD_win", opts={"title": "errD"})
errG_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="errG_win", opts={"title": "errG"})

text_win = vis.text('start')

global_step = torch.tensor([0])

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        global_step += 1
        input = data.cuda()  # .to(device) #todo [0]
        n = min(input.shape[0], 8)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real

        netD.zero_grad()
        # real_cpu = data  # , _
        # batch_size = real_cpu.size(0)
        # input.data.resize_(real_cpu.size()).copy_(real_cpu)
        # label.data.resize_(real_cpu.size(0)).fill_(real_label)

        label = torch.full((input.shape[0], 1), real_label).cuda()  # .to(device)

        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()  # todo 等同于torch.mean(output) ?

        # train with fake
        noise = torch.empty((input.shape[0], nz, 1, 1)).normal_().cuda()  # .to(device)
        # noise.data.resize_(batch_size, nz, 1, 1)
        # noise.data.normal_(0, 1)
        gen = netG.decoder(noise)

        # showImg(input.detach()[:n].cpu(), gen.detach()[:n].cpu()) # * 0.5 + 0.5   * 0.5 + 0.5
        # tbw.add_images("{} epoch {} batch images".format(epoch, i), gen.detach()[:n].cpu(), global_step,
        #                dataformats='HW')
        label.data.fill_(fake_label)
        output = netD(gen.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()  #
        D_G_z1 = output.data.mean()  # todo output.mean()
        errD = errD_real + errD_fake
        ##########################################
        # if i % 10 == 0:
        optimizerD.step()
        ##########################################
        if i % opt.nlog == 0:
            vis.line(Y=errD.view(1), X=global_step, win=errD_win, update="append", opts={"title": "errD_win"})
            gen_win = vis.images(gen[:n].cpu() * 0.5 + 0.5, win=gen_win,
                                 opts={"title": "noise_gen"})  # gen torch.Size([64, 1, 64, 64])

        ############################
        # (2) Update G network: VAE
        ###########################

        netG.zero_grad()

        # encoded = netG.encoder(input)
        # mu = encoded[0]
        # logvar = encoded[1]
        #
        # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # KLD = torch.sum(KLD_element).mul_(-0.5)  # extract to function
        #
        # sampled = netG.sampler(encoded)
        # rec = netG.decoder(sampled)
        rec, mu_logvar = netG(input)
        # KLD = KLD_loss(mu_logvar)
        KLD = KLDLoss(mu_logvar[0], mu_logvar[1])  # / 2  # / 10
        # rec_win = vis.images(rec[:n].cpu() * 0.5 + 0.5, win=rec_win)
        #
        MSEerr = MSECriterion(rec, input)  # MSECriterion(rec, input)
        VAEerr = KLD + MSEerr  # kld * 7
        ######################################
        VAEerr = VAEerr
        ######################################
        VAEerr.backward()
        optimizerG.step()  # todo 暂时不更新
        ######################################
        if i % opt.nlog == 0:
            vis.line(Y=MSEerr.view(1), X=global_step, win=VAE_MSE_win, update="append",
                     opts={"title": "VAE_MSEerr_win"})
            vis.line(Y=KLD.view(1), X=global_step, win=VAE_KLD_win, update="append", opts={"title": "VAE_KLDerr_win"})
            # rec_win = vis.images(rec[:n].cpu() * 0.5 + 0.5, win=rec_win)  # gen torch.Size([64, 1, 64, 64])
            origin_rec_win = vis.images(torch.cat((input[:n], rec[:n])).cpu() * 0.5 + 0.5, win=origin_rec_win,
                                        opts={"title": "origin_rec"})

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        # netG.zero_grad()  # todo 应该有？
        # netD.zero_grad()

        label.data.fill_(real_label)  # fake labels are real for generator cost #todo real_label

        rec, _ = netG(input)  # this tensor is freed from mem at this point
        output = netD(rec)
        errG = criterion(output, label) * 50  # * 30
        D_G_z2 = output.data.mean()
        ######################################
        errG.backward()
        # if i % 5 == 0:
        optimizerG.step()
        #####################################
        if i % opt.nlog == 0:
            vis.line(Y=errG.view(1), X=global_step, win=errG_win, update="append", opts={"title": "errG_win"})  #

        # print('[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
        #       % (epoch, opt.niter, i, len(dataloader),
        #          VAEerr.item(), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % opt.nlog == 0:
            print('[%d/%d][%d/%d]' % (epoch, opt.niter, i, len(dataloader)))
            vis.text('[%d/%d][%d/%d]' % (epoch, opt.niter, i, len(dataloader)), win=text_win,
                     opts={"title": "current epoch/batch"})
        if i % 10 * opt.nlog == 0:
            comparison = torch.cat([input.view(input.shape[0], 1, opt.imageSize, opt.imageSize)[-n:],
                                    rec.view(rec.shape[0], 1, opt.imageSize, opt.imageSize)[-n:]])
            save_image(comparison.cpu(),
                       'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            torch.save(netG.state_dict(),
                       '%s/netG_epoch_%d.pth' % (opt.outf, int(opt.nz)))  # todo 测试自动保存
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, int(opt.nz)))
        # if i % 10 == 0:
        #     global_step += 10
        #     origin_rec_win = vis.images(torch.cat((input[:n], rec[:n])).cpu() * 0.5 + 0.5, win=origin_rec_win)

    if (epoch + 1) % opt.saveInt == 0:  # and epoch != 0
        torch.save(netG.state_dict(),
                   '%s/netG_epoch_%d_%s.pth' % (opt.outf, epoch, str(int(time.time()))))  # todo 测试自动保存
        torch.save(netD.state_dict(), '%s/netD_epoch_%d_%s.pth' % (opt.outf, epoch, str(int(time.time()))))

# cv2.destroyAllWindows()
# tbw.close()
