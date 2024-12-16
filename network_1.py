import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EqualLR:
    def __init__(self, name):
        self.name = name
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * np.sqrt(2 / fan_in)


    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn
    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, pad, pixel_norm=True):
        super().__init__()

        self.kernel = kernel
        self.stride = 1
        self.pad = pad

        if pixel_norm:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, self.kernel, self.stride, self.pad),
                                    PixelNorm(),
                                    nn.LeakyReLU(0.2))
        else:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, self.kernel, self.stride, self.pad),
                                    nn.LeakyReLU(0.2))
    def forward(self, input):
        out = self.conv(input)
        return out

class ToGrayConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=1):
        super().__init__()

        self.kernel = 1
        self.stride = 1
        self.pad = 0

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, self.kernel, self.stride, self.pad),
                                nn.Tanh())
    def forward(self, input):
        out = self.conv(input)
        return out

class Decoder(nn.Module):
    def __init__(self, out_ch, code_dim=128):
        super().__init__()
        self.code_norm = PixelNorm()
        self.linear = nn.Linear(code_dim, 8192)
        self.init_togray = ToGrayConv(512, out_ch, 1)
        self.progression = nn.ModuleList([ConvBlock(512, 256, 3, 1),
                                          ConvBlock(256, 128, 3, 1),
                                          ConvBlock(128, 64, 3, 1),
                                          ConvBlock(64, 32, 3, 1),
                                          ConvBlock(32, 16, 3, 1),
                                          ConvBlock(16, 8, 3, 1)])
        self.to_gray = nn.ModuleList([ToGrayConv(256, out_ch, 1),
                                     ToGrayConv(128, out_ch, 1),
                                     ToGrayConv(64, out_ch, 1),
                                     ToGrayConv(32, out_ch, 1),
                                     ToGrayConv(16, out_ch, 1),
                                     ToGrayConv(8, out_ch, 1)])

    def forward(self, input, expand=0, alpha=-1):
        if expand==0:
            out = self.code_norm(input)
            out = self.linear(out).view(-1, 512, 4, 4)
            out = self.init_togray(out)
        else:
            out = self.code_norm(input)
            out = self.linear(out).view(-1, 512, 4, 4)
            for i, (conv, to_gray) in enumerate(zip(self.progression, self.to_gray)):
                upsample = F.interpolate(out, scale_factor=2)
                out = conv(upsample)

                if i+1 == expand:
                    out = to_gray(out)
                    if i > 0 and 0 <= alpha < 1:
                        skip_rgb = self.to_gray[i - 1](upsample)
                        out = (1 - alpha) * skip_rgb + alpha * out
                    break
        return out

class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.init_conv = nn.Conv2d(in_ch, 512, 1)
        self.progression = nn.ModuleList([ConvBlock(8, 16, 3, 1, pixel_norm=False),
                                          ConvBlock(16, 32, 3, 1, pixel_norm=False),
                                          ConvBlock(32, 64, 3, 1, pixel_norm=False),
                                          ConvBlock(64, 128, 3, 1, pixel_norm=False),
                                          ConvBlock(128, 256, 3, 1, pixel_norm=False),
                                          ConvBlock(257, 512, 3, 1, pixel_norm=False)])
        self.from_gray = nn.ModuleList([nn.Conv2d(in_ch, 8, 1),
                                       nn.Conv2d(in_ch, 16, 1),
                                       nn.Conv2d(in_ch, 32, 1),
                                       nn.Conv2d(in_ch, 64, 1),
                                       nn.Conv2d(in_ch, 128, 1),
                                       nn.Conv2d(in_ch, 256, 1),
                                       nn.Conv2d(in_ch, 512, 1)])
        self.n_layer = len(self.progression)
        self.linear = nn.Sequential(nn.Linear(8192, 1),
                                    nn.Sigmoid())

    def forward(self, input, expand=0, alpha=-1):
        b, c, _, _ = input.size()
        if expand ==0:
            out = self.init_conv(input)
            out = self.linear(out.contiguous().view(b, -1))
        else:
            for i in range(expand, 0, -1):
                index = self.n_layer - i
                if i == expand:
                    out = self.from_gray[index](input)
                if i == 1:
                    mean_std = input.std(0).mean()
                    mean_std = mean_std.expand(input.size(0), 1, 8, 8)
                    out = torch.cat([out, mean_std], 1)
                out = self.progression[index](out)

                if i > 0:
                    out = F.avg_pool2d(out, 2)
                    if i == expand and 0 <= alpha < 1:
                        skip_rgb = F.avg_pool2d(input, 2)
                        skip_rgb = self.from_gray[index + 1](skip_rgb)
                        out = (1 - alpha) * skip_rgb + alpha * out
            out = self.linear(out.contiguous().view(b, -1))
        return out

class Encoder(nn.Module):
    def __init__(self, in_ch, code_dim=128):
        super().__init__()
        self.init_conv = nn.Conv2d(in_ch, 512, 1)
        self.progression = nn.ModuleList([ConvBlock(8, 16, 3, 1, pixel_norm=False),
                                          ConvBlock(16, 32, 3, 1, pixel_norm=False),
                                          ConvBlock(32, 64, 3, 1, pixel_norm=False),
                                          ConvBlock(64, 128, 3, 1, pixel_norm=False),
                                          ConvBlock(128, 256, 3, 1, pixel_norm=False),
                                          ConvBlock(256, 512, 3, 1, pixel_norm=False)])
        self.from_gray = nn.ModuleList([nn.Conv2d(in_ch, 8, 1),
                                       nn.Conv2d(in_ch, 16, 1),
                                       nn.Conv2d(in_ch, 32, 1),
                                       nn.Conv2d(in_ch, 64, 1),
                                       nn.Conv2d(in_ch, 128, 1),
                                       nn.Conv2d(in_ch, 256, 1),
                                       nn.Conv2d(in_ch, 512, 1)])
        self.n_layer = len(self.progression)
        self.linear = nn.Linear(8192, code_dim)

    def forward(self, input, expand=0, alpha=-1):
        b, c, _, _ = input.size()
        # input.resize_(b, c, 2**(expand+2), 2**(expand+2))
        if expand ==0:
            out = self.init_conv(input)
            out = self.linear(out.contiguous().view(b, -1))
        else:
            for i in range(expand, 0, -1):
                index = self.n_layer - i
                if i == expand:
                    out = self.from_gray[index](input)
                out = self.progression[index](out)

                if i > 0:
                    out = F.avg_pool2d(out, 2)
                    if i == expand and 0 <= alpha < 1:
                        skip_rgb = F.avg_pool2d(input, 2)
                        skip_rgb = self.from_gray[index + 1](skip_rgb)
                        out = (1 - alpha) * skip_rgb + alpha * out
            out = self.linear(out.contiguous().view(b, -1))
        return out
    
class Latent_Dis(nn.Module):
    def __init__(self, latent_size):
        super(Latent_Dis, self).__init__()
        self.fc = nn.Sequential(nn.Linear(latent_size, 1500),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(1500, 1),
                                nn.Sigmoid())

    def forward(self, latent_vec):
        
        out =  self.fc(latent_vec)
        return out