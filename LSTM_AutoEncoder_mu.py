import torch
from torch import nn
from myDataset import CreateVim2HiddenLoader, CreateMvMnistHiddenDataset, CreateMvMnistHiddenSeqLoader
import torch.nn.functional as F
import os, time
from vae_model import _netG
import numpy as np
import visdom

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda")

# Hyper Parameters
EPOCH = 20  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 20  # rnn time step #/ image height
ENCODER_INPUT_SIZE = 32  # rnn input size #/ image width
DECODER_INPUT_SIZE = 128
HIDDEN_SIZE = DECODER_INPUT_SIZE
Z_SIZE = 32
LR = 0.005  # learning rate
MU_SIZE = 32
LOGVAR_SIZE = 32

torch.manual_seed(2)

# log_dir = './runs/lstm_' + str(int(time.time()))
# tbw = SummaryWriter(log_dir=log_dir)
vis = visdom.Visdom(server='http://172.18.29.70', env='lstm_autoencoder')
assert vis.check_connection()

origin_rec_win = vis.images(torch.zeros(16, 1, 64, 64, dtype=torch.float).cpu(), opts={"title": "origin rec images"})
lstm_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="lstm_loss_win", opts={"title": "lstm_loss_win"})


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
# logvar_rnn = RNN().to(device)
# rnn = nn.DataParallel(rnn)
print(mu_rnn)
# print(logvar_rnn)


def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        nn.init.uniform_(m.weight)
        # if m.bias:
        #     # m.weight.data.fill_(0.01)
        #     m.bias.data.fill_(0.01)


mu_rnn.apply(init_weights)
# logvar_rnn.apply(init_weights)

# vae = VAE(64, 64, 1024, 0).to(device)
# load_model_path = "/data1/home/guangjie/Project/python/pixel-models/savedModels/cnn-vae-mv-mnist-1554796229-1024-e10-ok.pt"
# vae.load_state_dict(torch.load(load_model_path))
# print('load model --- ' + load_model_path)
model = _netG(64, 1, 64, 32, 1)
model.to(device)
model.load_state_dict(
    torch.load("/data1/home/guangjie/Project/python/pixel-models/outf/netG_epoch_8_1555159728.pth"))
print('load model --- ')
print(model)

mu_optimizer = torch.optim.Adam(mu_rnn.parameters(), lr=LR)  # optimize all cnn parameters
# logvar_optimizer = torch.optim.Adam(logvar_rnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
# loss_func = nn.KLDivLoss(reduction='batchmean')
# vim2hidDataLoader = CreateVim2HiddenLoader(
#     "/data1/home/guangjie/Data/vim-2-gallant/features/vae_rec_mu_logvar_2048.hdf5", TIME_STEP, batch_size=BATCH_SIZE)

data_loader = CreateMvMnistHiddenSeqLoader("/data1/home/guangjie/Data/MNIST/mv_mnist_z_vae_dcgan_round.hdf5",
                                           batch_size=BATCH_SIZE,
                                           num_workers=4)


def KLD_loss(input, target):
    # kld_loss = loss_func(rec_mu_logvar, mu_logvar)
    kld_loss = F.kl_div(input, target, reduction='batchmean')
    # all_loss = []
    # for rec, ori in zip(rec_mu_logvar, mu_logvar):
    #     loss = loss_func(rec, ori)
    #     all_loss.append(loss)
    # ret_loss = all_loss[0]
    # for loss in all_loss[1:]:
    #     ret_loss += loss
    return - kld_loss  # / 100


def KLDLoss(mu, logvar):
    # mu logvar shape = (batch_size, time_step, mu/logvar)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def KLDLossSeq(mu, logvar):
    klds = []
    for i in range(mu.shape[1]):
        _mu = mu[:, i, :]
        _logvar = logvar[:, i, :]
        kld = -0.5 * torch.sum(1 + _logvar - _mu.pow(2) - _logvar.exp())
        klds.append(kld)
    return torch.mean(torch.stack(klds))


def BCE_loss(rec_mu_logvar, mu_logvar):
    bce_loss = F.binary_cross_entropy(rec_mu_logvar, mu_logvar)
    return bce_loss


MSECriteria = nn.MSELoss(reduction='sum')


def MSELossSeq(input, target):
    batch_loss = []
    weight = input.shape[1]
    for i in range(input.shape[1]):
        _input = input[:, i, :]
        _target = target[:, i, :]
        mse = MSECriteria(_input, _target) * 2 * (weight - i)
        # #todo 通过加权重来指定正向重建还是反向重建
        batch_loss.append(mse)
    # for rec, orig in zip(input, target):
    #     loss = F.mse_loss(rec, orig, reduction='sum')
    #     batch_loss.append(loss)
    mse_loss = torch.mean(torch.stack(batch_loss))
    # mse_loss = F.mse_loss(rec_mu_logvar, mu_logvar)
    return mse_loss


global_step = torch.tensor([0])
######################## 分别训练mu 和 logvar #############################
for epoch in range(EPOCH):
    for step, _data in enumerate(data_loader):  # gives batch data
        data = _data.to(device)
        # data = _data[:, :, :MU_SIZE] #_data[:, :, :MU_SIZE].to(device)
        # b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        output = mu_rnn(data[:, 0, :, :])  # data[:, 1, :, :]
        # mu:   logvar: data[:, 1, :, :] rnn output input shape = torch.Size([64, 20, 32])
        kld_loss = KLDLossSeq(output, data[:, 1, :, :])  # cross entropy loss output[:, :, MU_SIZE:]
        mse_loss = MSELossSeq(output, data[:, 0, :, :])  # BCE_loss(output, data)
        # bce_loss = BCE_loss(output, data[:, 0, :, :])
        # kld_loss = KLD_loss(output, data[:, 0, :, :])
        loss = mse_loss + kld_loss  # bce_loss  # mse_loss #+ kld_loss  # + mse_loss  # bce_loss
        mu_optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        mu_optimizer.step()  # apply gradients
        if (step + 1) % 30 == 0:  # todo test loader

            # tbw.add_scalar('LSTM_AutoEncoder/train/loss', loss.item(), global_step)
            # tbw.add_scalar('LSTM_AutoEncoder/train/kld_loss', kld_loss.item(), global_step)
            # tbw.add_scalar('LSTM_AutoEncoder/train/mse_loss', mse_loss.item(), global_step)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())

            n = min(data.size(0), 8)
            ########################################################################
            # origin_z = model.sampler([data[:n, 0, :], data[:n, 0, MU_SIZE:]])
            origin_z = model.sampler([data[0, 0, ::2, :], data[0, 1, ::2, :]])  # shape = (batch_size,mu/var,seq,)
            # # .reparameterize(data[:n, 0, :1024], data[:n, 0, 1024:])
            origin = model.decoder(origin_z[:, :, np.newaxis, np.newaxis])

            rec_z = model.sampler([output[0, ::2, :], data[0, 1, ::2, :]])
            # model.sampler([output[:n, 0, :], _data[:n, 0, MU_SIZE:]])
            # rec_z = model.sampler([data[:n, 0, :MU_SIZE], data[:n, 0, MU_SIZE:]])
            rec = model.decoder(rec_z[:, :, np.newaxis, np.newaxis])

            vis.images(torch.cat((origin, rec)).cpu() * 0.5 + 0.5, nrow=origin.shape[0], win=origin_rec_win,
                       opts={"title": "origin_rec"})
            vis.line(Y=loss.view(1), X=global_step, win=lstm_loss_win, update="append",
                     opts={"title": "VAE_MSEerr_win"})
            global_step += 1
            # comparison = torch.cat([origin, rec])
            # save_image(comparison.cpu(),
            #            'lstm_results/reconstruction_' + str(epoch) + "_" + str(step) + '.png', nrow=n)

torch.save(mu_rnn.state_dict(),
           'lstm_outf/lstm_autoencoder_mu_' + str(HIDDEN_SIZE) + '_' + str(int(time.time())) + '.pth')
print('end')
# if step % 50 == 0:
#     test_output = rnn(test_x.to(device)).to('cpu')  # (samples, time_step, input_size)
#     pred_y = torch.max(test_output, 1)[1].data.numpy()
#     accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
#     print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
