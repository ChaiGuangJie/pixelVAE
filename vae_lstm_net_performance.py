import torch
from torch import nn
from myDataset import CreateVim2HiddenLoader, CreateMvMnistHiddenDataset, CreateMvMnistHiddenSeqLoader
import torch.nn.functional as F
import os, time
from vae_model import _netG
import numpy as np
import visdom

vis = visdom.Visdom(server='http://172.18.29.70', env='vae_lstm_union')
assert vis.check_connection()
# Hyper Parameters
EPOCH = 20  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 10
TIME_STEP = 20  # rnn time step #/ image height
ENCODER_INPUT_SIZE = 32  # rnn input size #/ image width
DECODER_INPUT_SIZE = 128
HIDDEN_SIZE = DECODER_INPUT_SIZE
Z_SIZE = 32
LR = 0.005  # learning rate
MU_SIZE = 32
LOGVAR_SIZE = 32

torch.manual_seed(2)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda")


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_cell = nn.LSTMCell(ENCODER_INPUT_SIZE, HIDDEN_SIZE)
        self.decoder_cell = nn.LSTMCell(DECODER_INPUT_SIZE, HIDDEN_SIZE)
        self.z = nn.Linear(HIDDEN_SIZE, Z_SIZE)
        # self.mu = nn.Linear(HIDDEN_SIZE, MU_SIZE)
        # self.logvar = nn.Linear(HIDDEN_SIZE, LOGVAR_SIZE)

    def forward(self, x):
        hx, cx = torch.full((BATCH_SIZE, HIDDEN_SIZE), 0.1), torch.full((BATCH_SIZE, HIDDEN_SIZE), 0.1)
        for i in range(TIME_STEP):
            if i == 0:
                hx, cx = self.encoder_cell(x[:, i, :])  # todo 第一帧时就给hx,cx
            else:
                hx, cx = self.encoder_cell(x[:, i, :], (hx, cx))  # todo hx,cx 应该只有第一帧为空【不只是12帧的第一帧】，后面都用上一次的值
            # out.append(hx)
        output = []
        # output_seq = torch.empty((TIME_STEP, BATCH_SIZE, HIDDEN_SIZE), requires_grad=True).to(device)
        for i in range(TIME_STEP):
            if i == 0:
                hx, cx = self.decoder_cell(torch.full(hx.shape, 0.1).to(device), (hx, cx))
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


mu_rnn = RNN().to(device)
print(mu_rnn)
mu_rnn.load_state_dict(
    torch.load("/data1/home/guangjie/Project/python/pixel-models/lstm_outf/lstm_autoencoder_mu_128_1555400528_ok.pth"))
print('model mu_rnn loaded!')

logvar_rnn = RNN().to(device)
print(logvar_rnn)
logvar_rnn.load_state_dict(torch.load(
    "/data1/home/guangjie/Project/python/pixel-models/lstm_outf/lstm_autoencoder_logvar_128_1555398266_ok.pth"))
print('model logvar loaded!')

netG = _netG(64, 1, 64, 32, 1).to(device)
print(netG)
netG.load_state_dict(
    torch.load("/data1/home/guangjie/Project/python/pixel-models/outf/netG_epoch_8_1555159728.pth"))
print('load model netG!')

data_loader = CreateMvMnistHiddenSeqLoader("/data1/home/guangjie/Data/MNIST/mv_mnist_z_vae_dcgan_round.hdf5",
                                           batch_size=BATCH_SIZE,
                                           num_workers=4)

origin_rec_win = vis.images(torch.zeros(20, 1, 64, 64, dtype=torch.float).cpu(), opts={"title": "origin rec images"})
text_win = vis.text('start')
for i, data in enumerate(data_loader):
    data = data.to(device)
    mu = data[:, 0, :, :]
    logvar = data[:, 1, :, :]

    rec_mu = mu_rnn(mu)
    rec_logvar = logvar_rnn(logvar)

    for seq in range(20):
        origin_z = netG.sampler([mu[:, seq, :], logvar[:, seq, :]])
        origin = netG.decoder(origin_z[:, :, np.newaxis, np.newaxis])

        rec_z = netG.sampler([rec_mu[:, seq, :], rec_logvar[:, seq, :]])
        rec = netG.decoder(rec_z[:, :, np.newaxis, np.newaxis])

        vis.images(torch.cat((origin, rec)).cpu() * 0.5 + 0.5, nrow=origin.shape[0], win=origin_rec_win,
                   opts={"title": "origin_rec"})
        text_win = vis.text(str(i),win=text_win)
        time.sleep(0.2)
    print(i)
