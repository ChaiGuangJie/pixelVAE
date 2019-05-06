import torch.nn as nn
import torch
from myDataset import Create_vim2_fmri_z_loader, Create_Vim2_fmri_Dataloader, Sampled_z_and_fmri_Dataset, \
    get_vim2_fmri_mean_std
import os
import visdom
import time
import h5py
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
        nn.init.uniform_(m.weight, a=-0.1, b=0.1)
        # if m.bias:
        nn.init.uniform_(m.bias, a=-0.1, b=0.1)
        # m.weight.data.normal_(0.0, 0.02)
        # m.bias.data.normal_(0.0, 0.02)


def fmri_mapping_to_sampled_z(vis, voxel_file, train_z_file, test_z_file, seq_index=0, input_dim=4917, output_dim=512,
                              epochs=30,
                              lr=0.0001,
                              init_weights=init_weights, subject=2):
    model = LinearRegressionModel(input_dim, output_dim).cuda()
    model.apply(init_weights)

    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent

    dataLoader = Create_vim2_fmri_z_loader(voxel_file=voxel_file, z_file=train_z_file, dt_key='rt', seq_index=seq_index,
                                           batch_size=32)
    test_DataLoader = Create_vim2_fmri_z_loader(voxel_file=voxel_file, z_file=test_z_file, dt_key='rv',
                                                seq_index=seq_index,
                                                batch_size=32)

    loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="MSE Loss", opts={"title": "MSE Loss"})
    test_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="Test MSE Loss",
                             opts={"title": "Test MSE Loss"})
    text_win = vis.text('start')

    global_step = torch.tensor([0])
    test_global_step = torch.tensor([0])

    for ep in range(epochs):
        for step, (fmriData, zData) in enumerate(dataLoader):
            # (fmriData, zData)
            model.zero_grad()
            fmri = fmriData.cuda()
            out = model(fmri)
            # todo 归一化？

            loss = criterion(out, zData.cuda())
            loss.backward()
            optimiser.step()
            if step % 40 == 0:
                print('epoch/step : %d / %d, train loss : %.4f' % (ep, step, loss))
                vis.line(Y=loss.view(1), X=global_step, win=loss_win, update="append", opts={"title": "MSE Loss"})
                vis.text('[%d/%d][%d/%d],lr:%.4f' % (ep, epochs, step, len(dataLoader), lr), win=text_win,
                         opts={"title": "current epoch/batch"})
                global_step += 1
        with torch.no_grad():
            all_loss = []
            for step, (fmriData, zData) in enumerate(test_DataLoader):
                fmri = fmriData.cuda()
                out = model(fmri)
                loss = criterion(out, zData.cuda())
                all_loss.append(loss)

            mean_loss = sum(all_loss) / len(all_loss)
            vis.line(Y=mean_loss.view(1), X=test_global_step, win=test_loss_win, update="append",
                     opts={"title": "Test MSE Loss"})
            test_global_step += 1

    torch.save(model.state_dict(),
               'regression_outf/subject%d_regression_model_frame_%s_epoch_%d_%d.pth' % (subject,
                                                                                        str(seq_index), epochs,
                                                                                        output_dim))


def fmri_to_z(vis, voxel_file, train_z_file, test_z_file, regressor='mu', input_dim=4917, output_dim=1024, epochs=30,
              lr=0.001,
              init_weights=init_weights):
    model = LinearRegressionModel(input_dim, output_dim).cuda()

    model.apply(init_weights)

    criterion = nn.MSELoss()  # Mean Squared Loss
    optimiser = torch.optim.SGD(model.parameters(), lr=lr)  # Stochastic Gradient Descent

    dataLoader = Create_vim2_fmri_z_loader(voxel_file=voxel_file, z_file=train_z_file, dt_key='rt')
    test_DataLoader = Create_vim2_fmri_z_loader(voxel_file=voxel_file,
                                                z_file=test_z_file, dt_key='rv')

    loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="MSE Loss", opts={"title": "MSE Loss"})
    test_loss_win = vis.line(torch.tensor([0]), torch.tensor([0]), name="Test MSE Loss",
                             opts={"title": "Test MSE Loss"})
    text_win = vis.text('start')

    global_step = torch.tensor([0])
    test_global_step = torch.tensor([0])

    for ep in range(epochs):
        for step, (fmriData, zData) in enumerate(dataLoader):
            # (fmriData, zData)
            model.zero_grad()
            fmri = fmriData.cuda()
            out = model(fmri)
            # todo 归一化？

            if regressor == 'mu':
                input = zData[:, 0, :].cuda()
            elif regressor == 'logvar':  # logvar
                input = zData[:, 1, :].cuda()
            else:
                input = zData.cuda()

            loss = criterion(out, input)
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
                # mu = zData[:, 0, :].cuda()
                # logvar = zData[:, 1, :].cuda()
                # out = model(fmri)
                # if regressor == 'mu':
                #     loss = criterion(out, mu)  # logvar mu
                # else:  # logvar
                #     loss = criterion(out, logvar)
                out = model(fmri)
                if regressor == 'mu':
                    input = zData[:, 0, :].cuda()
                elif regressor == 'logvar':  # logvar
                    input = zData[:, 1, :].cuda()
                else:
                    input = zData.cuda()
                loss = criterion(out, input)
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


def apply_fmri_to_sampled_z(preTrainedModel, voxel_file, save_file, dt_key='rt', input_dim=4917, output_dim=512):
    model = LinearRegressionModel(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(preTrainedModel))

    dataLoader = Create_Vim2_fmri_Dataloader(voxel_file=voxel_file, dt_key=dt_key, shuffle=False, num_workers=1)

    with h5py.File(save_file, 'w') as sf:
        reg_z = sf.create_dataset(name='z', shape=(len(dataLoader.dataset), output_dim))
        begin_idx = 0
        with torch.no_grad():
            for step, fmriData in enumerate(dataLoader):
                fmri = fmriData  # .cuda()  # nan to num
                z = model(fmri)
                end_idx = begin_idx + len(fmriData)
                reg_z[begin_idx:end_idx, :] = z.detach().cpu().numpy()
                begin_idx = end_idx
                print(step)


def apply_regression_to_fmri(preTrainedRegressionMu, preTrainedRegressionLogvar, voxel_file, save_file, dt_key='rt',
                             input_dim=4917, output_dim=1024):
    '''
    将 fmri 映射到 z
    :return:
    '''
    mu_model = LinearRegressionModel(input_dim, output_dim).cuda()
    mu_model.load_state_dict(torch.load(preTrainedRegressionMu))

    logvar_model = LinearRegressionModel(input_dim, output_dim).cuda()
    logvar_model.load_state_dict(torch.load(preTrainedRegressionLogvar))

    dataLoader = Create_Vim2_fmri_Dataloader(voxel_file=voxel_file, dt_key=dt_key, shuffle=False, num_workers=1)

    with h5py.File(save_file, 'w') as sf:
        reg_z = sf.create_dataset(name='z', shape=(2, len(dataLoader.dataset), output_dim))
        begin_idx = 0
        with torch.no_grad():
            for step, fmriData in enumerate(dataLoader):
                fmri = fmriData.cuda()  # nan to num

                reg_mu = mu_model(fmri)
                reg_logvar = logvar_model(fmri)

                end_idx = begin_idx + len(fmriData)
                reg_z[0, begin_idx:end_idx, :] = reg_mu.detach().cpu().numpy()
                reg_z[1, begin_idx:end_idx, :] = reg_logvar.detach().cpu().numpy()
                begin_idx = end_idx
                print(step)


def show_regression_performance(viz, ori_z_file, regression_z_file, seq_index=0, time_step=15, z_size=512):
    line_win = viz.line(Y=np.random.rand(z_size), opts=dict(showlegend=True))
    X = np.linspace(0, z_size, z_size)
    with h5py.File(ori_z_file, 'r') as orif:
        ori_z = orif['z'][seq_index::time_step, :]
        with h5py.File(regression_z_file, 'r') as regf:
            reg_z = regf['z']
            assert ori_z.shape == reg_z.shape
            for (oz, rz) in zip(ori_z, reg_z):
                print(oz.shape)
                print(rz.shape)
                line_win = viz.line(
                    X=np.column_stack((
                        X, X
                    )),
                    Y=np.column_stack((
                        oz, rz
                    )),
                    win=line_win,
                    opts={
                        'dash': np.array(['solid', 'solid']),  # 'dash'  'dashdot'
                        'linecolor': np.array([
                            [0, 191, 255],
                            [255, 0, 0],
                        ]),
                        'title': 'Different line dash types'
                    }
                )


if __name__ == '__main__':
    vis = visdom.Visdom(server='http://172.18.29.70', env='linear regression')
    assert vis.check_connection()

    # fmri_to_z(vis, voxel_file="/data1/home/guangjie/Data/vim-2-gallant/orig/VoxelResponses_subject1.mat",
    #           train_z_file="/data1/home/guangjie/Data/vim-2-gallant/features/vae_gan_st_sampled_z_512_first_frame.hdf5",
    #           test_z_file="/data1/home/guangjie/Data/vim-2-gallant/features/vae_gan_sv_sampled_z_512_first_frame.hdf5",
    #           regressor='sampled_z', output_dim=512, epochs=15)
    # for i in range(15):
    #     fmri_mapping_to_sampled_z(vis,
    #                               voxel_file="/data1/home/guangjie/Data/vim-2-gallant/orig/VoxelResponses_subject1.mat",
    #                               train_z_file="/data1/home/guangjie/Data/vim-2-gallant/features/vae_gan_st_sampled_z_1024.hdf5",
    #                               test_z_file="/data1/home/guangjie/Data/vim-2-gallant/features/vae_gan_sv_sampled_z_1024.hdf5",
    #                               seq_index=i, output_dim=1024, epochs=60, lr=0.1, subject=1)
    # seq_index = 0
    # show_regression_performance(vis,
    #                             ori_z_file="/data1/home/guangjie/Data/vim-2-gallant/features/vae_gan_st_sampled_z_380.hdf5",
    #                             regression_z_file="/data1/home/guangjie/Data/vim-2-gallant/features/fmri_mapping_to_vae_gan_z/subject1_rt_frame_" + str(
    #                                 seq_index) + "_380.hdf5",
    #                             seq_index=seq_index, z_size=380
    #                             )
    for i in range(15):
        apply_fmri_to_sampled_z(
            preTrainedModel="/data1/home/guangjie/Project/python/pixel-models/regression_outf/subject1_regression_model_frame_" + str(
                i) + "_epoch_60_1024.pth",
            voxel_file="/data1/home/guangjie/Data/vim-2-gallant/orig/VoxelResponses_subject1.mat",
            save_file="/data1/home/guangjie/Data/vim-2-gallant/features/fmri_mapping_to_vae_gan_z/subject1_rv_frame_" + str(
                i) + "_1024.hdf5",
            dt_key='rv', output_dim=1024)
    # apply_regression_to_fmri(
    #     preTrainedRegressionMu="/data1/home/guangjie/Project/python/pixel-models/lstm_outf/regression_model_mu_epoch_15_1556955684.pth",
    #     preTrainedRegressionLogvar="/data1/home/guangjie/Project/python/pixel-models/lstm_outf/regression_model_logvar_epoch_15_1556955744.pth",
    #     voxel_file="/data1/home/guangjie/Data/vim-2-gallant/orig/VoxelResponses_subject1.mat",
    #     save_file="/data1/home/guangjie/Data/vim-2-gallant/features/subject1_rt_regression_to_vae_z_512.hdf5",
    #     dt_key='rt', output_dim=512)
    # apply_fmri_to_sampled_z(
    #     preTrainedModel="/data1/home/guangjie/Project/python/pixel-models/lstm_outf/regression_model_sampled_z_epoch_15_1556959134.pth",
    #     voxel_file="/data1/home/guangjie/Data/vim-2-gallant/orig/VoxelResponses_subject1.mat",
    #     save_file="/data1/home/guangjie/Data/vim-2-gallant/features/subject1_rt_regression_to_vae_z_512.hdf5",
    #     dt_key='rt')
    print('end')
