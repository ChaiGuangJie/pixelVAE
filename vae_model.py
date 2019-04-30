import torch
import torch.nn as nn
import models
import util
from torch.nn import functional as F
import os
from myDataset import CreateMvMnistHiddenDataset
from torchvision.utils import save_image
from torch.autograd import Variable
import math


class VAE(nn.Module):
    def __init__(self, H, W, zsize, depth=0, colors=1, out_channels=1):
        super().__init__()
        self.H = H
        self.W = W
        self.vae_encoder = models.ImEncoder(in_size=(H, W), zsize=zsize, use_bn=False, depth=depth,
                                            colors=colors)
        # todo self.vae_encoder 输出的隐变量是拼接在一起的！
        self.vae_decoder = models.ImDecoder(in_size=(H, W), zsize=zsize, use_bn=False, depth=depth,
                                            out_channels=out_channels)

        ##############UnUse##############
        # self.dfc = nn.Linear(zsize, 4096 * 16)
        # self.rsp = util.Reshape((16, 64, 64))
        # self.dblock = util.Block(16, 32, deconv=True)
        # self.convTrsp = nn.ConvTranspose2d(32, 1, 1)
        #################################

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.vae_encoder(x.view(-1, 1, self.H, self.W))
        z = self.reparameterize(mu, logvar)
        return self.vae_decoder(z), mu, logvar


class LinearVAE(nn.Module):
    def __init__(self, H, W, zsize):
        super().__init__()
        self.H = H
        self.W = W
        self.efc1 = nn.Linear(self.H * self.W, 4096)
        self.efc2 = nn.Linear(4096, 2048)
        self.fc_mu = nn.Linear(2048, zsize)
        self.fc_logvar = nn.Linear(2048, zsize)

        self.dfc1 = nn.Linear(zsize, 2048)
        self.dfc2 = nn.Linear(2048, 4096)
        self.dfc3 = nn.Linear(4096, self.H * self.W)

    def vae_encoder(self, x):
        e = self.efc1(x)  # 无pooling .view(-1, self.H *self.W)
        e = F.relu(self.efc2(e))
        mu = self.fc_mu(e)
        logvar = torch.exp(self.fc_logvar(e))
        return mu, logvar

    def vae_decoder(self, z):
        d = F.relu(self.dfc1(z))
        d = F.relu(self.dfc2(d))
        return torch.sigmoid(self.dfc3(d))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.vae_encoder(x.view(-1, self.H * self.W))
        z = self.reparameterize(mu, logvar)
        return self.vae_decoder(z).view(-1, 1, self.H, self.W), mu, logvar


class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()

    def forward(self, mu_logvar, cuda=True):
        mu = mu_logvar[0]
        logvar = mu_logvar[1]

        std = logvar.mul(0.5).exp_()  # calculate the STDEV
        if cuda:  # todo changed
            eps = torch.cuda.FloatTensor(std.size()).normal_()  # random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class _Encoder(nn.Module):
    def __init__(self, imageSize, ngf, nz, nc):
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
    def __init__(self, imageSize, ngpu, ngf, nz, nc):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(imageSize, ngf, nz, nc)  # todo changed
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


class _netD(nn.Module):
    def __init__(self, imageSize, ngpu, ngf, ndf, nc):
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


class RNNAutoEncoder(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, hidden_size, z_size, batch_size, time_step, device):
        super().__init__()

        self.encoder_input_size = encoder_input_size
        self.decoder_input_size = decoder_input_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.batch_size = batch_size
        self.time_step = time_step
        self.device = device

        self.encoder_cell = nn.LSTMCell(self.encoder_input_size, self.hidden_size)
        self.decoder_cell = nn.LSTMCell(self.decoder_input_size, self.hidden_size)
        self.z = nn.Linear(self.hidden_size, self.z_size)

        # self.mu = nn.Linear(HIDDEN_SIZE, MU_SIZE)
        # self.logvar = nn.Linear(HIDDEN_SIZE, LOGVAR_SIZE)

    def forward(self, x):
        hx, cx = torch.full((self.batch_size, self.hidden_size), 0.1), torch.full((self.batch_size, self.hidden_size), 0.1)
        for i in range(self.time_step):
            if i == 0:
                hx, cx = self.encoder_cell(x[:, i, :])  # todo 第一帧时就给hx,cx
            else:
                hx, cx = self.encoder_cell(x[:, i, :], (hx, cx))  # todo hx,cx 应该只有第一帧为空【不只是12帧的第一帧】，后面都用上一次的值
            # out.append(hx)
        output = []
        # output_seq = torch.empty((TIME_STEP, BATCH_SIZE, HIDDEN_SIZE), requires_grad=True).to(device)
        for i in range(self.time_step):
            if i == 0:
                hx, cx = self.decoder_cell(torch.full(hx.shape, 0.1).to(self.device), (hx, cx))
            else:
                hx, cx = self.decoder_cell(hx, (hx, cx))  # todo 这里的输入应该是什么？
            # mu = torch.tanh(self.mu(hx))
            # logvar = torch.tanh(self.logvar(hx))
            z = torch.tanh(self.z(hx))
            # mu = F.softmax(self.mu(hx), dim=1)  # F.log_softmax(self.mu(hx), dim=1)  #
            # logvar = F.softmax(self.logvar(hx), dim=1)  # F.log_softmax(self.logvar(hx), dim=1)  #
            # output_seq[i] = torch.cat((mu, logvar), 1)
            output.append(z)
            # output.append(torch.cat((mu, logvar), 1))
        # return output_seq.permute(1, 0, 2)
        return torch.stack(output).permute(1, 0, 2)  # shape = (16,20,2048) BATCH_SIZE,TIME_STEP,HIDDEN_SIZE
        # [::-1]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")
    batch_size = 64
    vae = LinearVAE(64, 64, 1024, 0).to(device)
    load_model_path = "/data1/home/guangjie/Project/python/pixel-models/savedModels/cnn-vae-mv-mnist-1554824573-1024-e40-ok.pt"
    vae.load_state_dict(torch.load(load_model_path))
    print('load model --- ' + load_model_path)

    data_loader = CreateMvMnistHiddenDataset("/data1/home/guangjie/Data/MNIST/mv_mnist_z.hdf5", batch_size=batch_size,
                                             num_workers=4)

    for step, data in enumerate(data_loader):
        if step % 10 == 0:
            print('step: ', step)
            data = data.to(device)
            n = min(data.size(0), batch_size)
            origin_z = vae.reparameterize(data[:n, 0, :1024], data[:n, 0, 1024:])
            origin = vae.decoder(origin_z)

            save_image(origin.view(batch_size, 1, 64, 64),
                       'lstm_results/sample_' + str(step) + '.png')

print('end')
