import torch
from torch import nn
from myDataset import CreateVim2HiddenLoader
import torch.nn.functional as F
import os, time
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda")

# Hyper Parameters
EPOCH = 30  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 1
TIME_STEP = 15  # rnn time step / image height
INPUT_SIZE = 4096  # rnn input size / image width
HIDDEN_SIZE = 4096
LR = 0.001  # learning rate
MU_SIZE = 2048
LOGVAR_SIZE = 2048

torch.manual_seed(2)

log_dir = './runs/lstm_' + str(int(time.time()))
tbw = SummaryWriter(log_dir=log_dir)


class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_cell = nn.LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
        self.decoder_cell = nn.LSTMCell(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mu = nn.Linear(HIDDEN_SIZE, MU_SIZE)
        self.logvar = nn.Linear(HIDDEN_SIZE, LOGVAR_SIZE)

    def forward(self, x):
        hx, cx = None, None
        for i in range(TIME_STEP):
            if i == 0:
                hx, cx = self.encoder_cell(x[:, i, :])
            else:
                hx, cx = self.encoder_cell(x[:, i, :], (hx, cx))  # todo hx,cx 应该只有第一帧为空【不只是12帧的第一帧】，后面都用上一次的值
            # out.append(hx)
        output = []
        for i in range(TIME_STEP):
            hx, cx = self.encoder_cell(hx, (hx, cx))  # todo 这里的输入应该是什么？
            mu = F.softmax(self.mu(hx), dim=1)  #F.log_softmax(self.mu(hx), dim=1)  #
            logvar =  F.softmax(self.logvar(hx), dim=1)  #F.log_softmax(self.logvar(hx), dim=1)  #
            output.append(torch.cat((mu, logvar), 1))
        return torch.stack(output).permute(1, 0, 2)


rnn = RNN().to(device)
# rnn = nn.DataParallel(rnn)
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
# loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
loss_func = nn.KLDivLoss(reduction='batchmean')
vim2hidDataLoader = CreateVim2HiddenLoader(
    "/data1/home/guangjie/Data/vim-2-gallant/features/vae_rec_mu_logvar_2048.hdf5", TIME_STEP, batch_size=BATCH_SIZE)


def KLD_loss(rec_mu_logvar, mu_logvar):
    # kld_loss = loss_func(rec_mu_logvar, mu_logvar)
    kld_loss = F.kl_div(rec_mu_logvar, mu_logvar, reduction='batchmean')
    # all_loss = []
    # for rec, ori in zip(rec_mu_logvar, mu_logvar):
    #     loss = loss_func(rec, ori)
    #     all_loss.append(loss)
    # ret_loss = all_loss[0]
    # for loss in all_loss[1:]:
    #     ret_loss += loss
    return kld_loss / 100


def BCE_loss(rec_mu_logvar, mu_logvar):
    bce_loss = F.binary_cross_entropy(rec_mu_logvar, mu_logvar, reduce=False)
    return bce_loss / 100


def MSE_loss(rec_mu_logvar, mu_logvar):
    batch_loss = []
    for rec, orig in zip(rec_mu_logvar, mu_logvar):
        loss = F.mse_loss(rec, orig, reduction='sum')
        batch_loss.append(loss)
    mse_loss = torch.mean(torch.stack(batch_loss))
    # mse_loss = F.mse_loss(rec_mu_logvar, mu_logvar)
    return mse_loss


global_step = 0

for epoch in range(EPOCH):
    for step, data in enumerate(vim2hidDataLoader):  # gives batch data
        data = data.to(device)
        # b_x = b_x.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        output = rnn(data)  # rnn output
        # kld_loss = KLD_loss(output, data)  # cross entropy loss
        mse_loss = MSE_loss(output, data)
        # bce_loss = BCE_loss(output, data)
        loss = mse_loss  # kld_loss + mse_loss  # bce_loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if step % 10 == 0:
            global_step += 1
            tbw.add_scalar('LSTM_AutoEncoder/train/loss', loss.item(), global_step)
            # tbw.add_scalar('LSTM_AutoEncoder/train/kld_loss', kld_loss.item(), global_step)
            tbw.add_scalar('LSTM_AutoEncoder/train/mse_loss', mse_loss.item(), global_step)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())

print('end')
# if step % 50 == 0:
#     test_output = rnn(test_x.to(device)).to('cpu')  # (samples, time_step, input_size)
#     pred_y = torch.max(test_output, 1)[1].data.numpy()
#     accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
#     print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
