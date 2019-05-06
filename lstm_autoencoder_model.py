import torch
from torch import nn


class LSTM_Autoencoder(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, hidden_size, time_step):
        super().__init__()

        self.encoder_cell = nn.LSTMCell(encoder_input_size, hidden_size).cuda()
        self.decoder_cell = nn.LSTMCell(decoder_input_size, hidden_size).cuda()
        self.z = nn.Linear(hidden_size, encoder_input_size)
        self.hidden_size = hidden_size
        self.time_step = time_step
        # self.batch_size = batch_size
        # self.mu = nn.Linear(HIDDEN_SIZE, MU_SIZE)
        # self.logvar = nn.Linear(HIDDEN_SIZE, LOGVAR_SIZE)

    def encoder(self, x):
        hx = torch.full((x.shape[0], self.hidden_size), 0.1).cuda()
        cx = torch.full((x.shape[0], self.hidden_size), 0.1).cuda()
        for i in range(self.time_step):
            hx, cx = self.encoder_cell(x[:, i, :], (hx, cx))  # todo hx,cx 应该只有一个场景的第一帧为空【不只是12帧的第一帧】，后面都用上一次的值
        return hx, cx

    def decoder(self, cx):  # todo 解码的时候只需要提供cx 而不能提供hx! 考虑替换为GRUcell 或者解码的第一个hx赋随机采样值
        output = []
        # output_seq = torch.empty((TIME_STEP, BATCH_SIZE, HIDDEN_SIZE), requires_grad=True).to(device)
        for i in range(self.time_step):
            if i == 0:
                hx, cx = self.decoder_cell(torch.full(cx.shape, 0.1).cuda(),
                                           (torch.tanh(cx), cx))  # todo 这里的 tanh 需不需要？
            else:
                hx, cx = self.decoder_cell(hx, (hx, cx))
            # mu = torch.tanh(self.mu(hx))
            # logvar = torch.tanh(self.logvar(hx))
            z = torch.tanh(self.z(hx))
            # mu = F.softmax(self.mu(hx), dim=1)  # F.log_softmax(self.mu(hx), dim=1)  #
            # logvar = F.softmax(self.logvar(hx), dim=1)  # F.log_softmax(self.logvar(hx), dim=1)  #
            # output_seq[i] = torch.cat((mu, logvar), 1)
            output.append(z)
            # output.append(torch.cat((mu, logvar), 1))
        # return output_seq.permute(1, 0, 2)
        return torch.stack(output).permute(1, 0, 2)

    def forward(self, x):
        hx, cx = self.encoder(x)
        rec_z_req = self.decoder(cx)
        return rec_z_req

        # hx, cx = torch.full((self.batch_size, self.hidden_size), 0.1), torch.full((self.batch_size, self.hidden_size),
        #                                                                           0.1)
        # for i in range(self.time_step):
        #     if i == 0:
        #         hx, cx = self.encoder_cell(x[:, i, :])  # todo 第一帧时就给hx,cx
        #     else:
        #         hx, cx = self.encoder_cell(x[:, i, :], (hx, cx))  # todo hx,cx 应该只有第一帧为空【不只是12帧的第一帧】，后面都用上一次的值
        # output = []
        # for i in range(self.time_step):
        #     if i == 0:
        #         hx, cx = self.decoder_cell(torch.full(hx.shape, 0.1).cuda(), (hx, cx))
        #     else:
        #         hx, cx = self.decoder_cell(hx, (hx, cx))  # todo 这里的输入应该是什么？
        #     z = torch.tanh(self.z(hx))
        #     output.append(z)
        # return torch.stack(output).permute(1, 0, 2)
        # # [::-1]


class GRU_Autoencoder(nn.Module):
    def __init__(self, encoder_input_size, decoder_input_size, hidden_size, time_step):
        super().__init__()
        self.encoder_cell = nn.GRUCell(encoder_input_size, hidden_size).cuda()
        self.decoder_cell = nn.GRUCell(decoder_input_size, hidden_size).cuda()
        self.z = nn.Linear(hidden_size, encoder_input_size).cuda()
        self.hidden_size = hidden_size
        self.time_step = time_step

    def encoder(self, x):
        hx = torch.full((x.shape[0], self.hidden_size), 0.1).cuda()
        for i in range(self.time_step):
            hx = self.encoder_cell(x[:, i, :], hx)
        return hx

    def decoder(self, hx):
        z_list = []
        for i in range(self.time_step):
            hx = self.decoder_cell(hx, hx)
            z = self.z(hx) #torch.tanh(self.z(hx))
            z_list.append(z)

        return torch.stack(z_list).permute(1, 0, 2)  # 这样交换维度可以？

    def forward(self, x):
        hx = self.encoder(x)
        return self.decoder(hx)


if __name__ == "__main__":
    encoder_input_size = 512
    decoder_input_size = 1024
    hidden_size = 1024
    time_step = 15
    rnn = LSTM_Autoencoder(encoder_input_size, decoder_input_size, hidden_size, time_step).cuda()
    # rnn = GRU_Autoencoder(encoder_input_size, decoder_input_size, hidden_size, time_step).cuda()
    x = torch.randn(10, 15, 512).cuda()
    output = rnn(x)
    print(output.shape)
