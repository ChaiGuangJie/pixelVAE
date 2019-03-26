import os, tqdm, random, pickle,time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Pad
import models, util
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.functional import cross_entropy, softmax

SEEDFRAC = 2

class Args():
    task = 'mnist'
    model = 'vae-straight'
    epochs = 10
    kernel_size = 7  # todo imagenet task change
    num_layers = 3  # Number of pixelCNN layers todo imagenet task change
    vae_depth = 0  # Depth of the VAE in blocks (in addition to the 3 default blocks)
    channels = 60  # todo
    batch_size = 32
    zsize = 32  # Size of latent space. todo 1024
    # latent_size = 32
    # embedding_size = 300
    limit = None
    lr = 0.001
    data_dir = './data'
    tb_dir = './runs/pixel' #todo 切换路径，以免覆盖
    loadPreModel = False
    # cache_dir = './cache
def draw_sample(seeds, decoder, pixcnn, zs, seedsize=(0, 0)):
    b, c, h, w = seeds.size()
    sample = seeds.clone()
    if torch.cuda.is_available():
        sample, zs = sample.cuda(), zs.cuda()
    sample, zs = Variable(sample), Variable(zs)
    cond = decoder(zs)
    for i in tqdm.trange(h):
        for j in range(w):
            if i < seedsize[0] and j < seedsize[1]:
                continue
            for channel in range(c):
                result = pixcnn(sample, cond)
                probs = softmax(result[:, :, channel, i, j]).data
                pixel_sample = torch.multinomial(probs, 1).float() / 255.
                sample[:, channel, i, j] = pixel_sample.squeeze()
    return sample

def saveModel(encoder, decoder, pixcnn):
    fileName = 'mod_' + str(int(time.time()))
    saveDir = 'savedModels/' + fileName
    os.mkdir(saveDir)
    torch.save(encoder.state_dict(),saveDir +'/encoder.pth')
    torch.save(decoder.state_dict(),saveDir +'/decoder.pth')
    torch.save(pixcnn.state_dict(),saveDir +'/pixcnn.pth')

    print('model saved!')

def loadModel(encoder, decoder, pixcnn, modDir=None, defaultPath = 'savedModels/'):
    if modDir is None:
        file_lists = os.listdir(defaultPath)
        if len(file_lists) == 0:
            print('No model loaded!')
            return encoder,decoder,pixcnn
        file_lists.sort(key=lambda fn:os.path.getmtime(defaultPath+fn) if os.path.isdir(defaultPath+fn) else 0)
        modDir = defaultPath + file_lists[-1] #default load latest model

    encoder.load_state_dict(torch.load(modDir+'/encoder.pth'))
    decoder.load_state_dict(torch.load(modDir+'/decoder.pth'))
    pixcnn.load_state_dict(torch.load(modDir+'/pixcnn.pth'))
    print('load ' + modDir)
    return encoder,decoder,pixcnn

def go(arg):
    tbw = SummaryWriter(log_dir=arg.tb_dir)
    if arg.task == 'mnist':
        transform = Compose([Pad(padding=2), ToTensor()])

        trainset = torchvision.datasets.MNIST(root=arg.data_dir, train=True,
                                              download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root=arg.data_dir, train=False,
                                             download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 1, 32, 32

    elif arg.task == 'imagenet64':
        transform = Compose([ToTensor()])
        trainset = torchvision.datasets.ImageFolder(root=arg.data_dir + os.sep + 'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch_size,
                                                  shuffle=True, num_workers=2)
        testset = torchvision.datasets.ImageFolder(root=arg.data_dir + os.sep + 'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch_size,
                                                 shuffle=False, num_workers=2)
        C, H, W = 3, 64, 64
    else:
        raise Exception('Task not recognized.')

    krn = arg.kernel_size
    pad = krn // 2
    OUTCN = 64
    encoder = models.ImEncoder(in_size=(H, W), zsize=arg.zsize, depth=arg.vae_depth, colors=C)
    # decoder = util.Lambda(lambda x: x)  # identity
    decoder = models.ImDecoder(in_size=(H, W), zsize=arg.zsize, depth=arg.vae_depth, out_channels=OUTCN)
    pixcnn = models.CGated((C, H, W), (arg.zsize,), arg.channels, num_layers=arg.num_layers, k=krn, padding=pad)
    #######################
    if options.loadPreModel:
        encoder, decoder, pixcnn = loadModel(encoder,decoder,pixcnn)
    ########################
    mods = [encoder, decoder, pixcnn]
    if torch.cuda.is_available():
        for m in mods:
            m.cuda()
    print('Constructed network', encoder, decoder, pixcnn)

    sample_zs = torch.randn(12, arg.zsize)
    sample_zs = sample_zs.unsqueeze(1).expand(12, 6, -1).contiguous().view(72, 1, -1).squeeze(1)
    # A sample of 144 square images with 3 channels, of the chosen resolution
    # (144 so we can arrange them in a 12 by 12 grid)
    sample_init_zeros = torch.zeros(72, C, H, W)
    sample_init_seeds = torch.zeros(72, C, H, W)
    sh, sw = H // SEEDFRAC, W // SEEDFRAC
    # Init second half of sample with patches from test set, to seed the sampling
    testbatch = util.readn(testloader, n=12)
    testbatch = testbatch.unsqueeze(1).expand(12, 6, C, H, W).contiguous().view(72, 1, C, H, W).squeeze(1)
    sample_init_seeds[:, :, :sh, :] = testbatch[:, :, :sh, :]

    params = []
    for m in mods:
        params.extend(m.parameters())
    optimizer = Adam(params, lr=arg.lr)
    instances_seen = 0
    for epoch in range(arg.epochs):
        # Train
        err_tr = []
        for m in mods:
            m.train(True)
        for i, (input, _) in enumerate(tqdm.tqdm(trainloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break
            # Prepare the input
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()
            target = (input.data * 255).long()
            input, target = Variable(input), Variable(target)
            # Forward pass
            zs = encoder(input)
            kl_loss = util.kl_loss(*zs)
            z = util.sample(*zs)
            out = decoder(z)
            rec = pixcnn(input, out)
            rec_loss = cross_entropy(rec, target, reduce=False).view(b, -1).sum(dim=1)
            loss = (rec_loss + kl_loss).mean()
            instances_seen += input.size(0)
            tbw.add_scalar('pixel-models/vae/training/kl-loss', kl_loss.mean().data.item(), instances_seen)
            tbw.add_scalar('pixel-models/vae/training/rec-loss', rec_loss.mean().data.item(), instances_seen)
            err_tr.append(loss.data.item())
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluate
        # - we evaluate on the test set, since this is only a simpe reproduction experiment
        #   make sure to split off a validation set if you want to tune hyperparameters for something important
        err_te = []

        for m in mods:
            m.train(False)
        for i, (input, _) in enumerate(tqdm.tqdm(testloader)):
            if arg.limit is not None and i * arg.batch_size > arg.limit:
                break
            b, c, w, h = input.size()
            if torch.cuda.is_available():
                input = input.cuda()
            target = (input.data * 255).long()
            input, target = Variable(input), Variable(target)

            zs = encoder(input)
            kl_loss = util.kl_loss(*zs)
            z = util.sample(*zs)
            out = decoder(z)
            rec = pixcnn(input, out)
            rec_loss = cross_entropy(rec, target, reduce=False).view(b, -1).sum(dim=1)
            loss = (rec_loss + kl_loss).mean()
            err_te.append(loss.data.item())

        tbw.add_scalar('pixel-models/test-loss', sum(err_te) / len(err_te), epoch)
        print('epoch={:02}; training loss: {:.3f}; test loss: {:.3f}'.format(
            epoch, sum(err_tr) / len(err_tr), sum(err_te) / len(err_te)))

        for m in mods:
            m.train(False)
        sample_zeros = draw_sample(sample_init_zeros, decoder, pixcnn, sample_zs, seedsize=(0, 0))
        sample_seeds = draw_sample(sample_init_seeds, decoder, pixcnn, sample_zs, seedsize=(sh, W))
        sample = torch.cat([sample_zeros, sample_seeds], dim=0)

        torchvision.utils.save_image(sample, 'myResults/sample_{:02d}.png'.format(epoch), nrow=12, padding=0)

    saveModel(encoder,decoder,pixcnn)

if __name__ == "__main__":
    options = Args()

    go(options)

    print('end')
