import torch
import torch.nn.functional as F
from torch import nn
from decoders.basic import ConvBnRelu, Conv1x1, ConvBNR, EdgeModule
from collections import OrderedDict

BatchNorm2d = nn.BatchNorm2d


class Boundary(nn.Module):
    def __init__(self, in_channels, inner_channels=256, k=10, bias=False, adaptive=False, smooth=False, serial=False,
                 *args, **kwargs):
        super(Boundary, self).__init__()
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4

        binary_channels = 256

        self.k = k
        self.serial = serial

        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)

        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )

        self.out_channels = self.conv_out

        self.conv3_1 = ConvBNR(64, 64, 3)
        self.dconv1 = ConvBNR(64, 64, 3, dilation=2)
        self.dconv2 = ConvBNR(64, 64, 3, dilation=3)
        self.dconv3 = ConvBNR(64, 64, 3, dilation=4)
        self.conv256x256 = Conv1x1(256, 256)

        self.edge = EdgeModule()

        self.binarize = nn.Sequential(
            nn.Conv2d(binary_channels, binary_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(binary_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(binary_channels // 4, binary_channels // 4, 2, 2),
            BatchNorm2d(binary_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(binary_channels // 4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                binary_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(*module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, x, gt=None, masks=None, training=False):
        x1, x2, x3, x4 = x

        p4 = self.reduce_conv_c5(x4)
        p3 = self._upsample_add(p4, self.reduce_conv_c4(x3))
        p3 = self.smooth_p4(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c3(x2))
        p2 = self.smooth_p3(p2)
        p1 = self._upsample_add(p2, self.reduce_conv_c2(x1))
        p1 = self.smooth_p2(p1)

        pall = self._upsample_cat(p1, p2, p3, p4)

        f1_1 = self._upsample_add(p2, p1)
        f1 = self.conv3_1(f1_1)

        f2_1 = self._upsample(p2, f1)
        f2_2 = self._upsample(p3, f1)
        f2 = self.dconv1(f1 + f2_1 + f2_2)

        f3_1 = self._upsample(p3, f2)
        f3_2 = self._upsample(p4, f2)
        f3 = self.dconv2(f2 + f3_1 + f3_2)

        f4_1 = self._upsample(p4, f3)
        f4 = self.dconv3(f3 + f4_1)

        fall = torch.cat([f1, f2, f3, f4], dim=1)

        p = self.conv256x256(pall + fall)

        t = self.edge(x1, x2, x3)
        t = torch.sigmoid(t)
        t = t * pall
        t = self.conv256x256(t)

        binary = self.binarize(p)
        if self.training:
            result = OrderedDict(binary=binary)
        else:
            return binary
        if self.adaptive and self.training:
            thresh = self.thresh(t)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:],mode='nearest') + y

    def _upsample(self, x, y):
        return F.interpolate(x, size=y.size()[2:],mode='nearest')

    def _upsample_concat(self, x, y):
        return torch.cat([F.interpolate(x, size=y.size()[2:],mode='nearest'), y], dim=1)

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w),mode='nearest')
        p4 = F.interpolate(p4, size=(h, w),mode='nearest')
        p5 = F.interpolate(p5, size=(h, w),mode='nearest')
        return torch.cat([p2, p3, p4, p5], dim=1)
