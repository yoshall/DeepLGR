import torch
import torch.nn as nn

class DeepLGR(nn.Module):
    def __init__(self, 
                 in_channels=2, # channels of each input frame
                 out_channels=2, # channels of each output frame
                 n_residuals=9, # number of SE blocks
                 n_filters=64, # number of filters in each SE block
                 t_params=(12, 3, 3),  # closeness, periodic, trend
                 height=128, # flow map height
                 width=128, # flow map width
                 pred_step=1, # number of ahead steps
                 flag_global=True, # whether to use global relation module
                 predictor='td' # "td": tensor decomposition, "md": matrix decomposition
                 ):

        super(DeepLGR, self).__init__()

        self.height = height
        self.width = width
        self.n_filters = n_filters
        self.out_channels = out_channels * pred_step
        self.flag_global = flag_global
        self.predictor = predictor

        in_channels = sum(t_params) * in_channels

        # SENet
        self.conv1 = nn.Conv2d(in_channels, n_filters, 3, 1, 1)
        se_blocks = []
        for _ in range(n_residuals):
            se_blocks.append(SEBlock(n_filters))
        self.senet = nn.Sequential(*se_blocks)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, 1, 1)

        if flag_global:
            self.glonet = GlobalNet(64, 64, (1, 2, 4, 8), height, width)
        
        # specify predictor

        if predictor == 'td': # tensor decomposition
            d1 = 16
            d2 = 16
            d3 = 32
            self.core = nn.Parameter(torch.FloatTensor(d1, d2, d3)) 
            self.F = nn.Parameter(torch.FloatTensor(d3, n_filters * self.out_channels)) 
            self.H = nn.Parameter(torch.FloatTensor(d1, height))
            self.W = nn.Parameter(torch.FloatTensor(d2, width))
            nn.init.normal_(self.core, 0, 0.02)
            nn.init.normal_(self.F, 0, 0.02)
            nn.init.normal_(self.H, 0, 0.02)
            nn.init.normal_(self.W, 0, 0.02)
        elif predictor == 'md': # matrix factorization
            self.L = nn.Parameter(torch.FloatTensor(height * width, 10))
            self.R = nn.Parameter(torch.FloatTensor(10, n_filters * self.out_channels))
            nn.init.normal_(self.L, 0, 0.02)
            nn.init.normal_(self.R, 0, 0.02)
        else:
            self.output_conv = nn.Sequential(nn.Conv2d(n_filters, self.out_channels, 1, 1, 0))

    def forward(self, inputs):
        # input: a tuple of three tensors regarding the closeness, periodic and trend.
        #        each of them are of shape [batch_size, in_channels * nb_previous_steps, height, weight]
        # 
        # output: [batch_size, out_channels * nb_future_steps, height, weight]
        # in our application, in_channels = out_channels = 2, i.e., inflow and outflow

        out = torch.cat(inputs, dim=1)
        b = out.shape[0]

        # senet blocks
        out1 = self.conv1(out)
        out = self.senet(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        # glonet
        if self.flag_global:
            out = self.glonet(out) # [b, n_filters, H, W]

        # predictor
        if self.predictor == 'td': # tensor decomposition
            out = out.reshape(b, self.n_filters, -1).permute(0, 2, 1) # [b, H*W, n_filters]
            region_param = torch.matmul(self.core, self.F) # [16, 16, n_filters*out_channels]
            region_param = region_param.permute(1, 2, 0) # [16, n_f*out_c, 16]
            region_param = torch.matmul(region_param, self.H) # [16, n_f*out_c, H]
            region_param = region_param.permute(1, 2, 0) # [n_f*out_c, H, 16]
            region_param = torch.matmul(region_param, self.W) # [n_f*out_c, H, W]
            region_param = region_param.unsqueeze(0).repeat(b, 1, 1, 1) # [b, n_f*out_c, H, W]
            region_param = region_param.reshape(b, -1, self.n_filters, self.height*self.width).permute(0, 3, 2, 1) # [b, H*W, n_f, 2]
            region_features = out.unsqueeze(3).repeat(1, 1, 1, self.out_channels) # [b, H*W, n_filters, 2]
            out = torch.sum(region_features * region_param, 2).reshape(b, self.height, self.width, -1)  # [b, H, W, 2]
            out = out.permute(0, 3, 1, 2) 
        elif self.predictor == 'md': # matrix decomposition
            out = out.reshape(b, self.n_filters, -1).permute(0, 2, 1) # [b, H*W, n_filters]
            region_param = torch.matmul(self.L, self.R).unsqueeze(0) # [1, H*W, n_filter*2]
            region_param = region_param.repeat(b, 1, 1).reshape(b, -1, self.n_filters, self.out_channels) # [b, H*W, n_filter, 2]
            region_features = out.unsqueeze(3).repeat(1, 1, 1, self.out_channels) # [b, H*W, n_filters, 2]
            out = torch.sum(region_features * region_param, 2).reshape(b, self.height, self.width, -1)  # [b, H, W, 2]
            out = out.permute(0, 3, 1, 2)
        else:
            out = self.output_conv(out)
        return out

class GlobalNet(nn.Module):
    def __init__(self, features=64, out_features=64, sizes=(1, 2, 4, 8), height=128, width=128):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features + features // 8 * 4, out_features, kernel_size=1)
        self.relu = nn.ReLU()

        self.deconvs = nn.ModuleList()
        for size in sizes:
            self.deconvs.append(SubPixelBlock(features // 8, upscale_factor=height // size))
        
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features // 8, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(features // 8)
        return nn.Sequential(prior, conv, bn)

    def forward(self, x):
        priors = [upsample(stage(x)) for stage, upsample in zip(self.stages, self.deconvs)]
        out = priors + [x]
        bottle = self.bottleneck(torch.cat(out, 1))
        return self.relu(bottle)

class SubPixelBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(SubPixelBlock, self).__init__()
        self.r = upscale_factor
        out_channels = in_channels * upscale_factor * upscale_factor
        self.conv = nn.Conv2d(in_channels,  out_channels, 1, 1, 0)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # x: [b, c, h, w]
        x = self.conv(x) # [b, c*r^2, h, w]
        out = self.ps(x) # [b, c, h*r, w*r]
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    def __init__(self, in_features):
        super(SEBlock, self).__init__()

        conv_block = [nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_features, in_features, 3, 1, 1),
                      nn.BatchNorm2d(in_features)]

        self.se = SELayer(in_features)
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.se(out)
        return x + out