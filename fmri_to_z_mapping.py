import torch.nn as nn
import torch
from myDataset import Create_vim2_fmri_z_loader, Create_Vim2_fmri_Dataloader
import os
import visdom
import time
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function

        out = self.linear(x)
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.uniform_(m.weight, a=0, b=0.1)
        # if m.bias:
        nn.init.uniform_(m.bias, a=0, b=0.1)
        # m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.normal_(0.0, 0.02)


def fmri_to_z(vis, regressor='mu', input_dim=4917, output_dim=1024, epochs=30, lr=0.001, init_weights=init_weights):
    model = LinearRegressionModel(input_dim, output_dim).cuda()

    model.apply(init_weights)

    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent

    dataLoader = Create_vim2_fmri_z_loader()
    test_DataLoader = Create_vim2_fmri_z_loader(
        z_file="/data1/home/guangjie/Data/vim-2-gallant/features/lstm_autoencoder_sv_z_1024.hdf5", train=False)

    loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="MSE Loss", opts={"title": "MSE Loss"})
    test_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="Test MSE Loss",
                             opts={"title": "Test MSE Loss"})
    text_win = vis.text('start')

    global_step = torch.tensor([0])
    test_global_step = torch.tensor([0])

    for ep in range(epochs):
        for step, (fmriData, zData) in enumerate(dataLoader):
            # (fmriData, zData)
            fmri = fmriData.cuda()
            mu = zData[:, 0, :].cuda()
            logvar = zData[:, 1, :].cuda()
            # todo 归一化？
            model.zero_grad()
            out = model(fmri)
            if regressor == 'mu':
                loss = criterion(out, mu)  # logvar mu
            else:  # logvar
                loss = criterion(out, logvar)
            loss.backward()
            optimiser.step()
            if step % 20 == 0:
                print('epoch/step : %d / %d, train loss : %.4f' % (ep, step, loss))
                vis.line(Y=loss.view(1), X=global_step, win=loss_win, update="append", opts={"title": "MSE Loss"})
                vis.text('[%d/%d][%d/%d],lr:%.4f' % (ep, epochs, step, len(dataLoader), lr), win=text_win,
                         opts={"title": "current epoch/batch"})
                global_step += 1

        with torch.no_grad():
            all_loss = []
            for step, (fmriData, zData) in enumerate(test_DataLoader):
                fmri = fmriData.cuda()
                mu = zData[:, 0, :].cuda()
                logvar = zData[:, 1, :].cuda()
                out = model(fmri)
                if regressor == 'mu':
                    loss = criterion(out, mu)  # logvar mu
                else:  # logvar
                    loss = criterion(out, logvar)
                all_loss.append(loss)

            mean_loss = sum(all_loss) / len(all_loss)
            vis.line(Y=mean_loss.view(1), X=test_global_step, win=test_loss_win, update="append",
                     opts={"title": "Test MSE Loss"})

            test_global_step += 1
            # print(step)

        # if (ep + 1) % 30 == 0:
        #     for g in optimiser.param_groups:
        #         l_rate /= 10
        #         g['lr'] = l_rate
    torch.save(model.state_dict(),
               'lstm_outf/regression_model_%s_epoch_%d_%s.pth' % (regressor, epochs, str(int(time.time()))))


def apply_regression_to_fmri(voxel_file, save_file, dt_key='rt', input_dim=4917, output_dim=1024):
    '''
    将 fmri 映射到 z
    :return:
    '''
    mu_model = LinearRegressionModel(input_dim, output_dim).cuda()
    mu_model.load_state_dict(torch.load(
        "/data1/home/guangjie/Project/python/pixel-models/lstm_outf/regression_model_mu_epoch_30_1556593702.pth"))

    logvar_model = LinearRegressionModel(input_dim, output_dim).cuda()
    logvar_model.load_state_dict(torch.load(
        "/data1/home/guangjie/Project/python/pixel-models/lstm_outf/regression_model_logvar_epoch_30_1556593972.pth"))

    dataLoader = Create_Vim2_fmri_Dataloader(voxel_file=voxel_file, dt_key=dt_key, shuffle=False)  # subject1

    with h5py.File(save_file, 'w') as sf:
        reg_z = sf.create_dataset(name='z', shape=(2, len(dataLoader.dataset), output_dim))
        begin_idx = 0
        with torch.no_grad():
            for step, fmriData in enumerate(dataLoader):
                fmri = fmriData.cuda() # nan to num

                reg_mu = mu_model(fmri)
                reg_logvar = logvar_model(fmri)

                end_idx = begin_idx + len(fmriData)
                reg_z[0, begin_idx:end_idx, :] = reg_mu.detach().cpu().numpy()
                reg_z[1, begin_idx:end_idx, :] = reg_logvar.detach().cpu().numpy()
                begin_idx = end_idx
                print(step)


if __name__ == '__main__':
    vis = visdom.Visdom(server='http://172.18.29.70', env='linear regression')
    assert vis.check_connection()

    # fmri_to_z(vis)
    apply_regression_to_fmri(voxel_file="/data1/home/guangjie/Data/vim-2-gallant/orig/VoxelResponses_subject1.mat",
                             save_file="/data1/home/guangjie/Data/vim-2-gallant/features/subject1_rt_regression_to_lstm_z_1024.hdf5",
                             dt_key='rt')
    print('end')
