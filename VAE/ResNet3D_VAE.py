import torch
import torch.nn as nn

import sys
sys.path.append('../')

from models.BaseModelClass import BaseModel

"""
Implementation based on the original paper https://arxiv.org/pdf/1810.11654.pdf
"""


class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32, norm="group"):
        super(GreenBlock, self).__init__()
        if norm == "batch":
            norm_1 = nn.BatchNorm3d(num_features=in_channels)
            norm_2 = nn.BatchNorm3d(num_features=in_channels)
        elif norm == "group":
            norm_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
            norm_2 = nn.GroupNorm(num_groups=8, num_channels=in_channels)

        self.layer_1 = nn.Sequential(
            norm_1,
            nn.ReLU())

        self.layer_2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            norm_2,
            nn.ReLU())

        self.conv_3 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3),
                                stride=1, padding=1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        y = self.conv_3(x)
        y = y + x
        return y


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                              #stride=2, padding=1)
                              stride=(1,2,2), padding=(1,1,1))

    def forward(self, x):
        return self.conv(x)


class BlueBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32):
        super(BlueBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                              stride=(1,1,1), padding=(1,1,1))

    def forward(self, x):
        return self.conv(x)


class UpBlock1(nn.Module):
    """
    TODO fix transpose conv to double spatial dim
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock1, self).__init__()
        self.transp_conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                                              stride=2, padding=1)

    def forward(self, x):
        return self.transp_conv(x)


class UpBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock2, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1),
                                stride=1)
        # self.up_sample_1 = nn.Upsample(scale_factor=2, mode="bilinear") # TODO currently not supported in PyTorch 1.4 :(
        self.up_sample_1 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.up_sample_1(self.conv_1(x))


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, start_channels=32):
        super(ResNetEncoder, self).__init__()

        self.start_channels = start_channels
        self.down_channels_1 = 2 * self.start_channels
        self.down_channels_2 = 2 * self.down_channels_1
        self.down_channels_3 = 2 * self.down_channels_2
        # print("self.down_channels_3", self.down_channels_3)

        self.blue_1 = BlueBlock(in_channels=in_channels, out_channels=self.start_channels)

        self.drop = nn.Dropout3d(0.2)

        self.green_1 = GreenBlock(in_channels=self.start_channels)

        self.down_1 = DownBlock(in_channels=self.start_channels, out_channels=self.down_channels_1)

        self.green_2_1 = GreenBlock(in_channels=self.down_channels_1)
        self.green_2_2 = GreenBlock(in_channels=self.down_channels_1)

        self.down_2 = DownBlock(in_channels=self.down_channels_1, out_channels=self.down_channels_2)

        self.green_3_1 = GreenBlock(in_channels=self.down_channels_2)
        self.green_3_2 = GreenBlock(in_channels=self.down_channels_2)

        self.down_3 = DownBlock(in_channels=self.down_channels_2, out_channels=self.down_channels_3)

        self.green_4_1 = GreenBlock(in_channels=self.down_channels_3)
        self.green_4_2 = GreenBlock(in_channels=self.down_channels_3)
        self.green_4_3 = GreenBlock(in_channels=self.down_channels_3)
        self.green_4_4 = GreenBlock(in_channels=self.down_channels_3)

    def forward(self, x):
        x = self.blue_1(x)
        #print('en, x:', x.shape)

        x = self.drop(x)
        x1 = self.green_1(x)
        x = self.down_1(x1)

        x = self.green_2_1(x)
        x2 = self.green_2_2(x)
        x = self.down_2(x2)

        x = self.green_3_1(x)
        x3 = self.green_3_2(x)
        x = self.down_3(x3)

        x = self.green_4_1(x)
        x = self.green_4_2(x)
        x = self.green_4_3(x)
        x4 = self.green_4_4(x)
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, in_channels=256, classes=4):
        super(Decoder, self).__init__()
        out_up_1_channels = int(in_channels / 2)
        out_up_2_channels = int(out_up_1_channels / 2)
        out_up_3_channels = int(out_up_2_channels / 2)

        self.up_1 = UpBlock2(in_channels=in_channels, out_channels=out_up_1_channels)

        self.green_1 = GreenBlock(in_channels=out_up_1_channels)

        self.up_2 = UpBlock2(in_channels=out_up_1_channels, out_channels=out_up_2_channels)

        self.green_2 = GreenBlock(in_channels=out_up_2_channels)

        self.up_3 = UpBlock2(in_channels=out_up_2_channels, out_channels=out_up_3_channels)

        self.green_3 = GreenBlock(in_channels=out_up_3_channels)

        self.blue = BlueBlock(in_channels=out_up_3_channels, out_channels=classes)

    def forward(self, x1, x2, x3, x4):
        x = self.up_1(x4)
        print('x:', x4.shape, x.shape, x3.shape)
        x = self.green_1(x + x3)
        x = self.up_2(x)
        x = self.green_2(x + x2)
        x = self.up_3(x)
        x = self.green_3(x + x1)

        y = self.blue(x)
        return y


class VAE(nn.Module):
    def __init__(self, in_channels=256, in_dim=(10, 10, 10), out_dim=(2, 64, 64, 64)):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modalities = out_dim[0]
        self.encoder_channels = 16  # int(in_channels >> 4)
        self.split_dim = int(self.in_channels / 2)

        # self.reshape_dim = (int(self.out_dim[1] / 16), int(self.out_dim[2] / 16), int(self.out_dim[3] / 16))
        # self.linear_in_dim = int(16 * (in_dim[0] / 2) * (in_dim[1] / 2) * (in_dim[2] / 2))

        self.reshape_dim = (int(self.out_dim[1] / self.encoder_channels), int(self.out_dim[2] / self.encoder_channels),
                            int(self.out_dim[3] / self.encoder_channels))

        self.linear_in_dim = int(self.encoder_channels * (in_dim[0] / 2) * (in_dim[1] / 2) * (in_dim[2] / 2))

        self.linear_vu_dim = self.encoder_channels * self.reshape_dim[0] * self.reshape_dim[1] * self.reshape_dim[2]

        channels_vup2 = int(self.in_channels / 2)  # 128
        channels_vup1 = int(channels_vup2 / 2)  # 64
        channels_vup0 = int(channels_vup1 / 2)  # 32

        group_1 = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        relu_1 = nn.ReLU()
        conv_1 = nn.Conv3d(in_channels=in_channels, out_channels=self.encoder_channels, stride=2, kernel_size=(3, 3, 3),
                           padding=1)

        self.VD = nn.Sequential(group_1, relu_1, conv_1)

        print('## self.linear_in_dim: ', self.linear_in_dim)
        self.linear_1 = nn.Linear(self.linear_in_dim, in_channels)

        # TODO VU layer here
        self.linear_vu = nn.Linear(channels_vup2, self.linear_vu_dim)
        relu_vu = nn.ReLU()
        VUup_block = UpBlock2(in_channels=self.encoder_channels, out_channels=self.in_channels)
        self.VU = nn.Sequential(relu_vu, VUup_block)

        self.Vup2 = UpBlock2(in_channels, channels_vup2)
        self.Vblock2 = GreenBlock(channels_vup2)

        self.Vup1 = UpBlock2(channels_vup2, channels_vup1)
        self.Vblock1 = GreenBlock(channels_vup1)

        self.Vup0 = UpBlock2(channels_vup1, channels_vup0)
        self.Vblock0 = GreenBlock(channels_vup0)

        self.Vend = BlueBlock(channels_vup0, self.modalities)

    def forward(self, x):
        x = self.VD(x)
        x = x.view(-1, self.linear_in_dim)
        x = self.linear_1(x)
        mu = x[:, :self.split_dim]
        logvar = x[:, self.split_dim:]
        y = reparametrize(mu, logvar)
        y = self.linear_vu(y)
        y = y.view(-1, self.encoder_channels, self.reshape_dim[0], self.reshape_dim[1], self.reshape_dim[2])
        y = self.VU(y)
        y = self.Vup2(y)
        y = self.Vblock2(y)
        y = self.Vup1(y)
        y = self.Vblock1(y)
        y = self.Vup0(y)
        y = self.Vblock0(y)
        dec = self.Vend(y)
        return dec, mu, logvar


class ResNet3dVAE(BaseModel):
    def __init__(self, data_size, classes=4, max_conv_channels=256):
        super(ResNet3dVAE, self).__init__()
        #vae_in_dim = (int(data_size['n_slice'] >> 3), int(data_size['h'] >> 3), int(data_size['w'] >> 3))
        #vae_out_dim = (data_size['c'], data_size['n_slice'], data_size['h'], data_size['w'])
        vae_in_dim = (int(data_size['c'] ), int(data_size['h'] >> 3), int(data_size['w'] >> 3))
        vae_out_dim = (data_size['n_slice'], data_size['c'], data_size['h'], data_size['w'])

        self.classes = classes
        #self.modalities = in_channels
        start_channels = 8  # int(max_conv_channels >> 3)

        self.encoder = ResNetEncoder(in_channels=data_size['n_slice'], start_channels=start_channels)
        #self.decoder = Decoder(in_channels=max_conv_channels, classes=classes)
        self.vae = VAE(in_channels=int(start_channels<<3), in_dim=vae_in_dim, out_dim=vae_out_dim)


        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(start_channels * 8, classes)

    def forward(self, x):
        #print('#### x 3d:', x.shape)
        x1, x2, x3, x4 = self.encoder(x)

        x = self.avgpool(x4)
        #print('x, x4:', x.shape, x4.shape)
        x = x.view(x.size(0), -1)
        #print('x:', x.shape)
        x = self.fc(x)
        #print('x:', x.shape)

        #y = self.decoder(x1, x2, x3, x4)
        vae_out, mu, logvar = self.vae(x4)
        #return y, vae_out, mu, logvar
        return vae_out, x

    def test(self):
        inp = torch.rand(1, self.modalities, self.dim[0], self.dim[1], self.dim[2])
        ideal = torch.rand(1, self.classes, self.dim[0], self.dim[1], self.dim[2])
        y, vae_out, mu, logvar = self.forward(inp)
        assert vae_out.shape == inp.shape, vae_out.shape
        assert y.shape == ideal.shape
        assert mu.shape == logvar.shape
        print("3D-RESNET VAE test OK!")


def test_enc_dec():
    model = ResNetEncoder(in_channels=2)
    input = torch.rand(1, 2, 80, 80, 80)
    #input = torch.rand(1, 2, 32, 160, 160)

    x1, x2, x3, x4 = model(input)
    print('#### input:', input.shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)

    model2 = Decoder()
    y = model2(x1, x2, x3, x4)
    print("out", y.shape)


def testVAE():
    input = torch.rand(1, 128, 10, 10, 10)
    model = VAE(in_channels=128, in_dim=(10, 10, 10), out_dim=(2, 128, 128, 128))
    #input = torch.rand(1, 128, 8, 20, 20)
    #model = VAE(in_channels=128, in_dim=(8, 20, 20), out_dim=(2, 32, 160, 160))
    out, mu, logvar = model(input)
    print('#### input:', input.shape)
    print("Done.\n Final out shape is: ", out.shape)

if __name__ == '__main__':
    test_enc_dec()
    testVAE()