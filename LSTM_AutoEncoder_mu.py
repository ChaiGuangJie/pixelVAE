import torch
from torch import nn
from myDataset import CreateVim2HiddenLoader, Vim2GrayDataset
# , CreateVim2HiddenLoader, CreateMvMnistHiddenDataset, CreateMvMnistHiddenSeqLoader
import torch.nn.functional as F
import os, time
from vae_model import _netG
import numpy as np
import visdom
import torchvision.transforms as transforms
from lstm_autoencoder_model import LSTM_Autoencoder, GRU_Autoencoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

# Hyper Parameters
EPOCH = 60  # 50 train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 15  # rnn time step #/ image height
ENCODER_INPUT_SIZE = 512  # rnn input size #/ image width
DECODER_INPUT_SIZE = 720
HIDDEN_SIZE = DECODER_INPUT_SIZE
Z_SIZE = 512
LR = 0.0001  # learning rate
MU_SIZE = 512
LOGVAR_SIZE = 512

torch.manual_seed(2)

# log_dir = './runs/lstm_' + str(int(time.time()))
# tbw = SummaryWriter(log_dir=log_dir)
vis = visdom.Visdom(server='http://172.18.29.70', env='rnn_autoencoder')
assert vis.check_connection()

origin_rec_win = vis.images(torch.zeros(16, 1, 64, 64, dtype=torch.float).cpu(), opts={"title": "origin rec images"})
lstm_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="RNN_loss_win", opts={"title": "RNN_loss_win"})

test_origin_rec_win = vis.images(torch.zeros(16, 1, 64, 64, dtype=torch.float).cpu(),
                                 opts={"title": "test origin rec images"})
test_lstm_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="test RNN_loss_win",
                              opts={"title": "test RNN_loss_win"})

# mu_rnn = LSTM_Autoencoder(ENCODER_INPUT_SIZE, DECODER_INPUT_SIZE, HIDDEN_SIZE, TIME_STEP).to(
#     device)
mu_rnn = GRU_Autoencoder(ENCODER_INPUT_SIZE, DECODER_INPUT_SIZE, HIDDEN_SIZE, TIME_STEP).to(
    device)
# RNN().to(device)
# logvar_rnn = RNN().to(device)
# rnn = nn.DataParallel(rnn)
print(mu_rnn)


# print(logvar_rnn)


def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        nn.init.uniform_(m.weight, 0, 0.1)
        nn.init.uniform_(m.bias, 0, 0.1)

    # elif type(m) == nn.RNNCellBase:
    #     nn.init.uniform_(m.weight_ih, 0, 0.1)
    #     nn.init.uniform_(m.weight_hh, 0, 0.1)
    #     nn.init.uniform_(m.bias_ih, 0, 0.1)
    #     nn.init.uniform_(m.bias_hh, 0, 0.1)
    # nn.init.uniform_(m.weight, 0, 0.1)
    # if m.bias:
    #     # m.weight.data.fill_(0.01)
    #     m.bias.data.fill_(0.01)


mu_rnn.apply(init_weights)
# logvar_rnn.apply(init_weights)

# vae = VAE(64, 64, 1024, 0).to(device)
# load_model_path = "/data1/home/guangjie/Project/python/pixel-models/savedModels/cnn-vae-mv-mnist-1554796229-1024-e10-ok.pt"
# vae.load_state_dict(torch.load(load_model_path))
# print('load model --- ' + load_model_path)
model = _netG(64, 1, 128, Z_SIZE, 1)
model.to(device)
model.load_state_dict(
    torch.load("/data1/home/guangjie/Project/python/pixel-models/outf/netG_epoch_20_1556167798_512_ok_0_7.pth"))
print('load model --- ')
print(model)

mu_optimizer = torch.optim.Adam(mu_rnn.parameters(), lr=LR)  # optimize all cnn parameters
# logvar_optimizer = torch.optim.Adam(logvar_rnn.parameters(), lr=LR)
# loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
# loss_func = nn.KLDivLoss(reduction='batchmean')
# vim2hidDataLoader = CreateVim2HiddenLoader(
#     "/data1/home/guangjie/Data/vim-2-gallant/features/vae_rec_mu_logvar_2048.hdf5", TIME_STEP, batch_size=BATCH_SIZE)

# data_loader = CreateMvMnistHiddenSeqLoader("/data1/home/guangjie/Data/MNIST/mv_mnist_z_vae_dcgan_round.hdf5",
#                                            batch_size=BATCH_SIZE,
#                                            num_workers=4)
data_loader = CreateVim2HiddenLoader("/data1/home/guangjie/Data/vim-2-gallant/features/vae_gan_st_z_512.hdf5",
                                     shuffle=True, num_workers=0)  # todo 不打乱会 loss 会规律下降

test_loader = CreateVim2HiddenLoader("/data1/home/guangjie/Data/vim-2-gallant/features/vae_gan_sv_z_512.hdf5",
                                     shuffle=True, num_workers=0)


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
test_global_step = torch.tensor([0])
######################## 分别训练mu 和 logvar #############################
for epoch in range(EPOCH):
    for step, _data in enumerate(data_loader):  # gives batch data
        data = _data.to(device)  # todo vae-gan 的z 和 rnn 输出的z 范围是否一致？
        mu_optimizer.zero_grad()  # clear gradients for this training step
        # data = _data[:, :, :MU_SIZE] #_data[:, :, :MU_SIZE].to(device)
        # b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        batch_mu = data[:, 0, :, :]  # shape = (batch_size, seq_len, z_size)
        batch_logvar = data[:, 1, :, :]  # shape = (batch_size, seq_len, z_size)
        output = mu_rnn(batch_mu)  # shape = (batch_size,seq_len,z_size)
        # mu:   logvar: data[:, 1, :, :] rnn output input shape = torch.Size([64, 20, 32])
        kld_loss = KLDLossSeq(output, batch_logvar)  # cross entropy loss output[:, :, MU_SIZE:]
        mse_loss = MSELossSeq(output, batch_mu)  # BCE_loss(output, data)
        # bce_loss = BCE_loss(output, data[:, 0, :, :])
        # kld_loss = KLD_loss(output, data[:, 0, :, :])
        loss = mse_loss + kld_loss  # bce_loss  # mse_loss #+ kld_loss  # + mse_loss  # bce_loss
        loss.backward()  # backpropagation, compute gradients
        mu_optimizer.step()  # apply gradients
        if (step + 1) % 30 == 0:
            # tbw.add_scalar('LSTM_AutoEncoder/train/loss', loss.item(), global_step)
            # tbw.add_scalar('LSTM_AutoEncoder/train/kld_loss', kld_loss.item(), global_step)
            # tbw.add_scalar('LSTM_AutoEncoder/train/mse_loss', mse_loss.item(), global_step)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())

            n = min(data.size(0), 8)
            ########################################################################
            # origin_z = model.sampler([data[:n, 0, :], data[:n, 0, MU_SIZE:]])
            origin_z = model.sampler([batch_mu[:n, 0, :], batch_logvar[:n, 0, :]])
            # shape = (batch_size,mu/var,seq,) [data[0, 0, ::2, :], data[0, 1, ::2, :]]
            # # .reparameterize(data[:n, 0, :1024], data[:n, 0, 1024:])
            origin = model.decoder(origin_z.view(n, origin_z.shape[-1], 1, 1))  # origin_z[:, :, np.newaxis, np.newaxis]

            rec_z = model.sampler([output[:n, 0, :], batch_logvar[:n, 0, :]])  # data[0, 1, ::2, :]
            # model.sampler([output[:n, 0, :], _data[:n, 0, MU_SIZE:]])
            # rec_z = model.sampler([data[:n, 0, :MU_SIZE], data[:n, 0, MU_SIZE:]])
            rec = model.decoder(rec_z.view(n, rec_z.shape[-1], 1, 1))  # rec_z[:, :, np.newaxis, np.newaxis]

            # vis.images(origin.cpu() * 0.5 + 0.5, nrow=origin.shape[0], win=origin_rec_win)
            grid = torch.cat((origin, rec))
            vis.images(grid.cpu() * 0.5 + 0.5, nrow=n, win=origin_rec_win,
                       opts={"title": "origin_rec"})
            vis.line(Y=loss.view(1), X=global_step, win=lstm_loss_win, update="append",
                     opts={"title": "RNN_loss_win"})
            global_step += 1

            # comparison = torch.cat([origin, rec])
            # save_image(comparison.cpu(),
            #            'lstm_results/reconstruction_' + str(epoch) + "_" + str(step) + '.png', nrow=n)
    with torch.no_grad():
        loss_list = []
        for step, data in enumerate(test_loader):
            data = data.to(device)
            batch_mu = data[:, 0, :, :]  # shape = (batch_size, seq_len, z_size)
            batch_logvar = data[:, 1, :, :]  # shape = (batch_size, seq_len, z_size)
            output = mu_rnn(batch_mu)  # shape = (batch_size,seq_len,z_size)

            kld_loss = KLDLossSeq(output, batch_logvar)  # cross entropy loss output[:, :, MU_SIZE:]
            mse_loss = MSELossSeq(output, batch_mu)  # BCE_loss(output, data)
            loss = mse_loss + kld_loss
            loss_list.append(loss)
            n = min(data.size(0), 8)
            origin_z = model.sampler([batch_mu[:n, 0, :], batch_logvar[:n, 0, :]])
            origin = model.decoder(origin_z.view(n, origin_z.shape[-1], 1, 1))

            rec_z = model.sampler([output[:n, 0, :], batch_logvar[:n, 0, :]])  # data[0, 1, ::2, :]
            rec = model.decoder(rec_z.view(n, rec_z.shape[-1], 1, 1))  # rec_z[:, :, np.newaxis, np.newaxis]

            grid = torch.cat((origin, rec))
            vis.images(grid.cpu() * 0.5 + 0.5, nrow=n, win=test_origin_rec_win,
                       opts={"title": "test origin rec images"})
        vis.line(Y=sum(loss_list).view(1) / len(loss_list), X=test_global_step, win=test_lstm_loss_win, update="append",
                 opts={"title": "test RNN_loss_win"})
        test_global_step += 1
    if (epoch + 1) % 30 == 0:
        for g in mu_optimizer.param_groups:
            LR /= 10
            g['lr'] = LR

torch.save(mu_rnn.state_dict(),
           'lstm_outf/lstm_autoencoder_mu_' + str(HIDDEN_SIZE) + '_' + str(int(time.time())) + '.pth')
print('end')
# if step % 50 == 0:
#     test_output = rnn(test_x.to(device)).to('cpu')  # (samples, time_step, input_size)
#     pred_y = torch.max(test_output, 1)[1].data.numpy()
#     accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
#     print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
