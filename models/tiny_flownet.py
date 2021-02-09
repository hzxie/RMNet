# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-09-05 20:14:54
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-11-05 14:58:28
# @Email:  cshzxie@gmail.com
#
# References:
# - https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py

import torch
import torch.nn.functional as F

import utils.helpers


class TinyFlowNet(torch.nn.Module):
    def __init__(self, cfg):
        super(TinyFlowNet, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable
        self.conv3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable
        self.conv4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable

        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable
        self.conv5_1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable

        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable
        self.conv6_1 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1, inplace=True))  # yapf: disable

        self.deconv5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=True),
            torch.nn.LeakyReLU(0.1, inplace=True))
        self.deconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1026, 256, kernel_size=4, stride=2, padding=1, bias=True),
            torch.nn.LeakyReLU(0.1, inplace=True))
        self.deconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(770, 128, kernel_size=4, stride=2, padding=1, bias=True),
            torch.nn.LeakyReLU(0.1, inplace=True))
        self.deconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(386, 64, kernel_size=4, stride=2, padding=1, bias=True),
            torch.nn.LeakyReLU(0.1, inplace=True))

        self.predict_flow6 = torch.nn.Conv2d(1024, 2, kernel_size=3, padding=1, bias=True)
        self.predict_flow5 = torch.nn.Conv2d(1026, 2, kernel_size=3, padding=1, bias=True)
        self.predict_flow4 = torch.nn.Conv2d(770, 2, kernel_size=3, padding=1, bias=True)
        self.predict_flow3 = torch.nn.Conv2d(386, 2, kernel_size=3, padding=1, bias=True)
        self.predict_flow2 = torch.nn.Conv2d(194, 2, kernel_size=3, padding=1, bias=True)

        self.upsampled_flow6_to_5 = torch.nn.ConvTranspose2d(2,
                                                             2,
                                                             kernel_size=4,
                                                             stride=2,
                                                             padding=1,
                                                             bias=False)
        self.upsampled_flow5_to_4 = torch.nn.ConvTranspose2d(2,
                                                             2,
                                                             kernel_size=4,
                                                             stride=2,
                                                             padding=1,
                                                             bias=False)
        self.upsampled_flow4_to_3 = torch.nn.ConvTranspose2d(2,
                                                             2,
                                                             kernel_size=4,
                                                             stride=2,
                                                             padding=1,
                                                             bias=False)
        self.upsampled_flow3_to_2 = torch.nn.ConvTranspose2d(2,
                                                             2,
                                                             kernel_size=4,
                                                             stride=2,
                                                             padding=1,
                                                             bias=False)

    def _forward(self, img0, img1):
        (img0, img1), pad = utils.helpers.pad_divide_by([img0, img1], 64,
                                                        (img0.size()[2], img0.size()[3]))

        out_conv2 = self.conv2(self.conv1(torch.cat((img0, img1), dim=1)))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), dim=1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), dim=1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), dim=1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), dim=1)
        flow2 = self.predict_flow2(concat2)
        flow2 = F.interpolate(flow2, scale_factor=4, mode='bilinear')

        if pad[2] + pad[3] > 0:
            flow2 = flow2[:, :, pad[2]:-pad[3], :]

        if pad[0] + pad[1] > 0:
            flow2 = flow2[:, :, :, pad[0]:-pad[1]]

        return flow2

    def forward(self, frames, device=None):
        batch_size, n_frames, _, h, w = frames.size()
        est_optical_flows = torch.zeros(batch_size, n_frames, 2, h, w).float()
        # Fix Assertion Error:  all(map(lambda i: i.is_cuda, inputs))
        # The value of device is set in utils/eval_server.py for full set evaluation
        if torch.cuda.device_count() > 1 and device is None:
            est_optical_flows = utils.helpers.var_or_cuda(est_optical_flows)

        for t in range(1, n_frames):
            est_optical_flows[:, t] = self._forward(frames[:, t], frames[:, t - 1])

        return est_optical_flows
