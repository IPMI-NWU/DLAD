import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
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

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None, pixel_norm=True):
#         super().__init__()

#         pad1 = padding
#         pad2 = padding
#         if padding2 is not None:
#             pad2 = padding2

#         kernel1 = kernel_size
#         kernel2 = kernel_size
#         if kernel_size2 is not None:
#             kernel2 = kernel_size2

#         convs = [EqualConv2d(in_channel, out_channel, kernel1, padding=pad1)]
#         if pixel_norm:
#             convs.append(PixelNorm())
#         convs.append(nn.LeakyReLU(0.1))
#         convs.append(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))
#         if pixel_norm:
#             convs.append(PixelNorm())
#         convs.append(nn.LeakyReLU(0.1))

#         self.conv = nn.Sequential(*convs)

#     def forward(self, input):
#         out = self.conv(input)
#         return out

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


def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)

def downscale(feat):
    return F.avg_pool2d(feat, 2)
    # return F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=False)


class Encoder(nn.Module):
    def __init__(self, in_ch=1, code_dim=128, pixel_norm=True):
        super().__init__()
        self.output_dim = code_dim

        self.progression_4 = nn.Linear(8192, code_dim)
        self.progression_8 = ConvBlock(256, 512, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(128, 256, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(64, 128, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(32, 64, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(16, 32, 3, 1, pixel_norm=pixel_norm)
        self.progression_256 = ConvBlock(8, 16, 3, 1, pixel_norm=pixel_norm)

        self.from_gray_4 = nn.Conv2d(in_ch, 512, 1)
        self.from_gray_8 = nn.Conv2d(in_ch, 256, 1)
        self.from_gray_16 = nn.Conv2d(in_ch, 128, 1)
        self.from_gray_32 = nn.Conv2d(in_ch, 64, 1)
        self.from_gray_64 = nn.Conv2d(in_ch, 32, 1)
        self.from_gray_128 = nn.Conv2d(in_ch, 16, 1)
        self.from_gray_256 = nn.Conv2d(in_ch, 8, 1)
        
        self.max_step = 6

    def progress(self, feat, module):
        out = module(feat)
        # out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)
        out = F.avg_pool2d(out, 2)
        return out

    def smooth_input(self, feat1, feat2, module1, alpha):

        skip_gray = module1(downscale(feat1))
        out = (1-alpha)*skip_gray + alpha*feat2
        return out

    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        if step== 0:
            out_4 = self.from_gray_4(input)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.progression_4(out_4)
            return [out]
        
        if step == 1:
            out_8 = self.from_gray_8(input)
            out_4 = self.progress(out_8, self.progression_8)
            if 0 <= alpha < 1:
                out_4 = self.smooth_input(input, out_4, self.from_gray_4, alpha)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.progression_4(out_4)
            return [out_4, out]
        
        if step == 2:
            out_16 = self.from_gray_16(input)
            out_8 = self.progress(out_16, self.progression_16)
            if 0 <= alpha < 1:
                out_8 = self.smooth_input(input, out_8, self.from_gray_8, alpha)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.progression_4(out_4)
            return  [out_8, out_4, out]

        if step == 3:
            out_32 = self.from_gray_32(input)
            out_16 = self.progress(out_32, self.progression_32)
            if 0 <= alpha < 1:
                out_16 = self.smooth_input(input, out_16, self.from_gray_16, alpha)
            out_8 = self.progress(out_16, self.progression_16)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.progression_4(out_4)
            return [out_16, out_8, out_4, out]
        
        if step == 4:
            out_64 = self.from_gray_64(input)
            out_32 = self.progress(out_64, self.progression_64)
            if 0 <= alpha < 1:
                out_32 = self.smooth_input(input, out_32, self.from_gray_32, alpha)
            out_16 = self.progress(out_32, self.progression_32)
            out_8 = self.progress(out_16, self.progression_16)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.progression_4(out_4)
            return [out_32, out_16, out_8, out_4, out]
        
        if step == 5:
            out_128 = self.from_gray_128(input)
            out_64 = self.progress(out_128, self.progression_128)
            if 0 <= alpha < 1:
                out_64 = self.smooth_input(input, out_64, self.from_gray_64, alpha)
            out_32 = self.progress(out_64, self.progression_64)
            out_16 = self.progress(out_32, self.progression_32)
            out_8 = self.progress(out_16, self.progression_16)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.progression_4(out_4)
            return [out_64, out_32, out_16, out_8, out_4, out]  

        if step == 6:
            out_256 = self.from_gray_256(input)
            out_128 = self.progress(out_256, self.progression_256)
            if 0 <= alpha < 1:
                out_128 = self.smooth_input(input, out_128, self.from_gray_128, alpha)
            out_64 = self.progress(out_128, self.progression_128)
            out_32 = self.progress(out_64, self.progression_64)
            out_16 = self.progress(out_32, self.progression_32)
            out_8 = self.progress(out_16, self.progression_16)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.progression_4(out_4)
            return [out_128, out_64, out_32, out_16, out_8, out_4, out]
            

class Decoder(nn.Module):
    def __init__(self, out_ch=1, code_dim=128, pixel_norm=True, tanh=True):
        super().__init__()
        self.input_dim = code_dim
        self.tanh = tanh

        self.progression_4 = nn.Linear(code_dim, 8192)
        self.progression_8 = ConvBlock(512, 256, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(256, 128, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(128, 64, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(64, 32, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(32, 16, 3, 1, pixel_norm=pixel_norm)
        self.progression_256 = ConvBlock(16, 8, 3, 1, pixel_norm=pixel_norm)

        self.to_gray_4 = nn.Conv2d(512, out_ch, 1)
        self.to_gray_8 = nn.Conv2d(256, out_ch, 1)
        self.to_gray_16 = nn.Conv2d(128, out_ch, 1)
        self.to_gray_32 = nn.Conv2d(64, out_ch, 1)
        self.to_gray_64 = nn.Conv2d(32, out_ch, 1)
        self.to_gray_128 = nn.Conv2d(16, out_ch, 1)
        self.to_gray_256 = nn.Conv2d(8, out_ch, 1)
        
        self.max_step = 6

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_gray = upscale(module1(feat1))
            out = (1-alpha)*skip_gray + alpha*module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step
            
        out_4 = self.progression_4(input)
        out_4 = out_4.view(-1, 512, 4, 4)
        if step==0:
            if self.tanh:
                return torch.tanh(self.to_gray_4(out_4))
            return self.to_gray_4(out_4)
        
        out_8 = self.progress(out_4, self.progression_8)
        if step==1:
            return self.output(out_4, out_8, self.to_gray_4, self.to_gray_8, alpha )
        
        out_16 = self.progress(out_8, self.progression_16)
        if step==2:
            return self.output( out_8, out_16, self.to_gray_8, self.to_gray_16, alpha )
        
        out_32 = self.progress(out_16, self.progression_32)
        if step==3:
            return self.output( out_16, out_32, self.to_gray_16, self.to_gray_32, alpha )

        out_64 = self.progress(out_32, self.progression_64)
        if step==4:
            return self.output( out_32, out_64, self.to_gray_32, self.to_gray_64, alpha )
        
        out_128 = self.progress(out_64, self.progression_128)
        if step==5:
            return self.output( out_64, out_128, self.to_gray_64, self.to_gray_128, alpha )

        out_256 = self.progress(out_128, self.progression_256)
        if step==6:
            return self.output( out_128, out_256, self.to_gray_128, self.to_gray_256, alpha )


class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.progression_8 = ConvBlock(257, 512, 3, 1, pixel_norm=False)
        self.progression_16 = ConvBlock(128, 256, 3, 1, pixel_norm=False)
        self.progression_32 = ConvBlock(64, 128, 3, 1, pixel_norm=False)
        self.progression_64 = ConvBlock(32, 64, 3, 1, pixel_norm=False)
        self.progression_128 = ConvBlock(16, 32, 3, 1, pixel_norm=False)
        self.progression_256 = ConvBlock(8, 16, 3, 1, pixel_norm=False)

        self.from_gray_4 = nn.Conv2d(in_ch, 512, 1)
        self.from_gray_8 = nn.Conv2d(in_ch, 256, 1)
        self.from_gray_16 = nn.Conv2d(in_ch, 128, 1)
        self.from_gray_32 = nn.Conv2d(in_ch, 64, 1)
        self.from_gray_64 = nn.Conv2d(in_ch, 32, 1)
        self.from_gray_128 = nn.Conv2d(in_ch, 16, 1)
        self.from_gray_256 = nn.Conv2d(in_ch, 8, 1)
        
        self.linear = nn.Sequential(nn.Linear(8192, 1),
                                    nn.Sigmoid())
        self.max_step = 6

    def progress(self, feat, module):
        out = module(feat)
        # out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)
        out = F.avg_pool2d(out, 2)
        return out

    def smooth_input(self, feat1, feat2, module1, alpha):

        skip_gray = module1(downscale(feat1))
        out = (1-alpha)*skip_gray + alpha*feat2
        return out

    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        if step== 0:
            out_4 = self.from_gray_4(input)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.linear(out_4)
            return out
        
        if step == 1:
            out_8 = self.from_gray_8(input)
            out_std = torch.sqrt(out_8.var(0, unbiased=False) + 1e-8)
            mean_std = out_std.mean()
            mean_std = mean_std.expand(out_8.size(0), 1, 8, 8)
            out_8 = torch.cat([out_8, mean_std], 1)
            out_4 = self.progress(out_8, self.progression_8)
            if 0 <= alpha < 1:
                out_4 = self.smooth_input(input, out_4, self.from_gray_4, alpha)    
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.linear(out_4)
            return out
        
        if step == 2:
            out_16 = self.from_gray_16(input)
            out_8 = self.progress(out_16, self.progression_16)
            if 0 <= alpha < 1:
                out_8 = self.smooth_input(input, out_8, self.from_gray_8, alpha)
            out_std = torch.sqrt(out_8.var(0, unbiased=False) + 1e-8)
            mean_std = out_std.mean()
            mean_std = mean_std.expand(out_8.size(0), 1, 8, 8)
            out_8 = torch.cat([out_8, mean_std], 1)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.linear(out_4)
            return out

        if step == 3:
            out_32 = self.from_gray_32(input)
            out_16 = self.progress(out_32, self.progression_32)
            if 0 <= alpha < 1:
                out_16 = self.smooth_input(input, out_16, self.from_gray_16, alpha)
            out_8 = self.progress(out_16, self.progression_16)
            out_std = torch.sqrt(out_8.var(0, unbiased=False) + 1e-8)
            mean_std = out_std.mean()
            mean_std = mean_std.expand(out_8.size(0), 1, 8, 8)
            out_8 = torch.cat([out_8, mean_std], 1)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.linear(out_4)
            return out
        
        if step == 4:
            out_64 = self.from_gray_64(input)
            out_32 = self.progress(out_64, self.progression_64)
            if 0 <= alpha < 1:
                out_32 = self.smooth_input(input, out_32, self.from_gray_32, alpha)
            out_16 = self.progress(out_32, self.progression_32)
            out_8 = self.progress(out_16, self.progression_16)
            out_std = torch.sqrt(out_8.var(0, unbiased=False) + 1e-8)
            mean_std = out_std.mean()
            mean_std = mean_std.expand(out_8.size(0), 1, 8, 8)
            out_8 = torch.cat([out_8, mean_std], 1)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.linear(out_4)
            return out
        
        if step == 5:
            out_128 = self.from_gray_128(input)
            out_64 = self.progress(out_128, self.progression_128)
            if 0 <= alpha < 1:
                out_64 = self.smooth_input(input, out_64, self.from_gray_64, alpha)
            out_32 = self.progress(out_64, self.progression_64)
            out_16 = self.progress(out_32, self.progression_32)
            out_8 = self.progress(out_16, self.progression_16)
            out_std = torch.sqrt(out_8.var(0, unbiased=False) + 1e-8)
            mean_std = out_std.mean()
            mean_std = mean_std.expand(out_8.size(0), 1, 8, 8)
            out_8 = torch.cat([out_8, mean_std], 1)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.linear(out_4)
            return out  

        if step == 6:
            out_256 = self.from_gray_256(input)
            out_128 = self.progress(out_256, self.progression_256)
            if 0 <= alpha < 1:
                out_128 = self.smooth_input(input, out_128, self.from_gray_128, alpha)
            out_64 = self.progress(out_128, self.progression_128)
            out_32 = self.progress(out_64, self.progression_64)
            out_16 = self.progress(out_32, self.progression_32)
            out_8 = self.progress(out_16, self.progression_16)
            out_std = torch.sqrt(out_8.var(0, unbiased=False) + 1e-8)
            mean_std = out_std.mean()
            mean_std = mean_std.expand(out_8.size(0), 1, 8, 8)
            out_8 = torch.cat([out_8, mean_std], 1)
            out_4 = self.progress(out_8, self.progression_8)
            out_4 = out_4.view(out_4.size(0), -1)
            out = self.linear(out_4)
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
