import torch
import torch.nn as nn
import models
import util

class VAE(nn.Module):
    def __init__(self,H,W,zsize,depth,colors=1,out_channels=1):
        super(VAE, self).__init__()
        self.vae_encoder = models.ImEncoder(in_size=(H, W), zsize=zsize, use_bn=False, depth=depth,
                                            colors=colors)
        # todo self.vae_encoder 输出的隐变量是拼接在一起的！
        self.decoder = models.ImDecoder(in_size=(H, W), zsize=zsize, use_bn=False, depth=depth,
                                        out_channels=out_channels)

        ##############UnUse##############
        self.dfc = nn.Linear(zsize, 4096 * 16)
        self.rsp = util.Reshape((16, 64, 64))
        self.dblock = util.Block(16, 32, deconv=True)
        self.convTrsp = nn.ConvTranspose2d(32, 1, 1)
        #################################

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.vae_encoder(x.view(-1, 1, 64, 64))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
