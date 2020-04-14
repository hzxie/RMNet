# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:07:00
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-14 16:50:57
# @Email:  cshzxie@gmail.com
#
# Maintainers:
# - Seoung Wug Oh <sw.oh@yonsei.ac.kr>
# - Anni Xu <xuanni@sensetime.com>
# - Yunmu Huang <huangyunmu@sensetime.com>
# - Haozhe Xie <cshzxie@gmail.com>

import math
import torch
import torch.nn.functional as F
import torchvision.models

import utils.helpers


class ResBlock(torch.nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim is None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = torch.nn.Conv2d(indim,
                                              outdim,
                                              kernel_size=3,
                                              padding=1,
                                              stride=stride)

        self.conv1 = torch.nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class EncoderMemory(torch.nn.Module):
    def __init__(self):
        super(EncoderMemory, self).__init__()
        self.conv1_m = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu    # 1/2, 64
        self.maxpool = resnet.maxpool
        self.res2 = resnet.layer1    # 1/4, 256
        self.res3 = resnet.layer2    # 1/8, 512
        self.res4 = resnet.layer3    # 1/8, 1024

    def forward(self, in_f, in_m, in_o):
        # print(in_f.shape)   # torch.Size([1, 3, 480, 864])
        # print(in_m.shape)   # torch.Size([1, 480, 864])
        # print(in_o.shape)   # torch.Size([1, 480, 864])
        m = torch.unsqueeze(in_m, dim=1).float()    # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float()    # add channel dim

        x = self.conv1(in_f) + self.conv1_m(m) + self.conv1_o(o)
        x = self.bn1(x)
        c1 = self.relu(x)    # 1/2, 64
        x = self.maxpool(c1)    # 1/4, 64
        r2 = self.res2(x)    # 1/4, 256
        r3 = self.res3(r2)    # 1/8, 512
        r4 = self.res4(r3)    # 1/8, 1024
        return r4, r3, r2, c1, in_f


class EncoderQuery(torch.nn.Module):
    def __init__(self):
        super(EncoderQuery, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu    # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1    # 1/4, 256
        self.res3 = resnet.layer2    # 1/8, 512
        self.res4 = resnet.layer3    # 1/8, 1024

    def forward(self, in_f):
        x = self.conv1(in_f)
        x = self.bn1(x)
        c1 = self.relu(x)    # 1/2, 64
        x = self.maxpool(c1)    # 1/4, 64
        r2 = self.res2(x)    # 1/4, 256
        r3 = self.res3(r2)    # 1/8, 512
        r4 = self.res4(r3)    # 1/8, 1024
        return r4, r3, r2, c1, in_f


class Refine(torch.nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = torch.nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(
            pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(torch.nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = torch.nn.Conv2d(1024, mdim, kernel_size=3, padding=1, stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)    # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim)    # 1/4 -> 1

        self.pred2 = torch.nn.Conv2d(mdim, 2, kernel_size=3, padding=1, stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)    # out: 1/8, 256
        m2 = self.RF2(r2, m3)    # out: 1/4, 256
        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p


class Memory(torch.nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, m_in, m_out, q_in, q_out):    # m_in: o,c,t,h,w
        B, D_e, T, H, W = m_in.size()
        _, D_o, _, _, _ = m_out.size()

        mi = m_in.view(B, D_e, T * H * W)
        mi = torch.transpose(mi, 1, 2)    # b, THW, emb

        qi = q_in.view(B, D_e, H * W)    # b, emb, HW

        p = torch.bmm(mi, qi)    # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)    # b, THW, HW

        mo = m_out.view(B, D_o, T * H * W)
        mem = torch.bmm(mo, p)    # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_out = torch.cat([mem, q_out], dim=1)

        return mem_out, p


class KeyValue(torch.nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x), self.value_conv(x)


class STM(torch.nn.Module):
    def __init__(self, cfg):
        super(STM, self).__init__()
        self.encoder_memory = EncoderMemory()
        self.encoder_query = EncoderQuery()
        self.kv_memory = KeyValue(1024, keydim=128, valdim=512)
        self.kv_query = KeyValue(1024, keydim=128, valdim=512)
        self.memory = Memory()
        self.decoder = Decoder(256)

    def pad_memory(self, mems, n_objects, K):
        pad_mems = []
        B = len(n_objects)

        for mem in mems:
            _, C, H, W = mem.size()
            pad_mem = utils.helpers.var_or_cuda(torch.zeros(B, K, C, 1, H, W))
            for i in range(B):
                begin = sum(n_objects[:i])
                end = begin + n_objects[i]
                pad_mem[i, 1:n_objects[i] + 1, :, 0] = mem[begin:end]

            pad_mems.append(pad_mem)

        return pad_mems

    def pad_divide_by(self, in_list, d, in_size):
        out_list = []
        h, w = in_size
        if h % d > 0:
            new_h = h + d - h % d
        else:
            new_h = h

        if w % d > 0:
            new_w = w + d - w % d
        else:
            new_w = w

        lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
        lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
        pad_array = (int(lw), int(uw), int(lh), int(uh))
        for inp in in_list:
            out_list.append(F.pad(inp, pad_array))

        return out_list, pad_array

    def memorize(self, frame, masks, n_objects):
        # memorize a frame
        B, K, H, W = masks.shape
        # print(frame.shape)    # torch.Size([1, 3, 480, 854])
        # print(masks.shape)    # torch.Size([1, n_objects + 1, 480, 854])
        (frame, masks), pad = self.pad_divide_by([frame, masks], 16,
                                                 (frame.size()[2], frame.size()[3]))
        # print(pad)            # (5, 5, 0, 0)

        # make batch arg list
        batch_list = {'f': [], 'm': [], 'o': []}
        for i in range(B):
            for o in range(1, n_objects[i] + 1):    # 1 - no
                batch_list['f'].append(frame[i].unsqueeze(0))
                batch_list['m'].append(masks[i, o].unsqueeze(0))
                batch_list['o'].append(
                    (torch.sum(masks[i, 1:o].unsqueeze(0), dim=1) +
                     torch.sum(masks[i, o + 1:n_objects[i] + 1].unsqueeze(0), dim=1)).clamp(0, 1))

        # make Batch
        for k, v in batch_list.items():
            batch_list[k] = torch.cat(v, dim=0)

        # print(batch_list['f'].shape)  # torch.Size([bs * n_objects, 3, 480, 864])
        # print(batch_list['m'].shape)  # torch.Size([bs * n_objects, 480, 864])
        # print(batch_list['o'].shape)  # torch.Size([bs * n_objects, 480, 864])
        r4, _, _, _, _ = self.encoder_memory(batch_list['f'], batch_list['m'], batch_list['o'])
        # print(r4.shape)       # torch.Size([bs * n_objects, 1024, 30, 54])
        k4, v4 = self.kv_memory(r4)    # n_objects, 128 and 512, H/16, W/16
        # print(k4.shape)       # torch.Size([bs * n_objects, 128, 30, 54])
        # print(v4.shape)       # torch.Size([bs * n_objects, 512, 30, 54])
        k4, v4 = self.pad_memory([k4, v4], n_objects=n_objects, K=K)
        # print(k4.shape)       # torch.Size([1, 11, 128, 1, 30, 54])
        # print(v4.shape)       # torch.Size([1, 11, 512, 1, 30, 54])
        return k4, v4

    def soft_aggregation(self, ps, K, n_objects):
        B = len(n_objects)
        _, H, W = ps.shape

        em = utils.helpers.var_or_cuda(torch.zeros(B, K, H, W))
        for i in range(B):
            begin = sum(n_objects[:i])
            end = begin + n_objects[i]
            em[i, 0] = torch.prod(1 - ps[begin:end], dim=0)    # bg prob
            em[i, 1:n_objects[i] + 1] = ps[begin:end]    # obj prob

        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def segment(self, frame, keys, values, n_objects):
        B, K, keydim, T, H, W = keys.shape
        # print(frame.shape)    # torch.Size([1, 3, 480, 854])
        [frame], pad = self.pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))
        # print(frame.shape)    # torch.Size([1, 3, 480, 854])
        # print(pad)            # (5, 5, 0, 0)
        r4, r3, r2, _, _ = self.encoder_query(frame)
        # print(r4.shape)       # torch.Size([1, 1024, 30, 54])
        # print(r3.shape)       # torch.Size([1, 512, 60, 108])
        # print(r2.shape)       # torch.Size([1, 256, 120, 216])
        k4, v4 = self.kv_query(r4)    # 1, dim, H/16, W/16
        # print(k4.shape)       # torch.Size([1, 128, 30, 54])
        # print(v4.shape)       # torch.Size([1, 512, 30, 54])
        batch_list = {
            'k4e': [],
            'v4e': [],
            'r3e': [],
            'r2e': [],
            'key': [],
            'value': [],
        }
        for i in range(B):
            # expand to ---  no, c, h, w
            _k4e, _v4e = k4[i].expand(n_objects[i], -1, -1,
                                      -1), v4[i].expand(n_objects[i], -1, -1, -1)
            _r3e, _r2e = r3[i].expand(n_objects[i], -1, -1,
                                      -1), r2[i].expand(n_objects[i], -1, -1, -1)
            _key = keys[i, 1:n_objects[i] + 1]
            _value = values[i, 1:n_objects[i] + 1]
            # print(_k4e.shape) # torch.Size([n_obj, 128, 30, 54])
            # print(_v4e.shape) # torch.Size([n_obj, 512, 30, 54])
            # print(_r3e.shape) # torch.Size([n_obj, 512, 60, 108])
            # print(_r2e.shape) # torch.Size([n_obj, 256, 120, 216])
            batch_list['k4e'].append(_k4e)
            batch_list['v4e'].append(_v4e)
            batch_list['r3e'].append(_r3e)
            batch_list['r2e'].append(_r2e)
            batch_list['key'].append(_key)
            batch_list['value'].append(_value)

        for k, v in batch_list.items():
            batch_list[k] = torch.cat(v, dim=0)

        # memory select kv:(1, K, C, T, H, W)
        # print(keys.shape)     # torch.Size([1, 11, 128, 1, 30, 54])
        # print(values.shape)   # torch.Size([1, 11, 512, 1, 30, 54])
        m4, viz = self.memory(batch_list['key'], batch_list['value'], batch_list['k4e'],
                              batch_list['v4e'])
        # print(m4.shape)       # torch.Size([1, 1024, 30, 54])
        # print(viz.shape)      # torch.Size([1, 3240, 1620])
        logits = self.decoder(m4, batch_list['r3e'], batch_list['r2e'])
        # print(logits.shape)   # torch.Size([1, 2, 480, 864])
        ps = F.softmax(logits, dim=1)    # no, h, w
        # print(ps.shape)       # torch.Size([1, 2, 480, 864])
        ps = ps[:, 1]
        # print(ps.shape)       # torch.Size([1, 480, 864])
        # ps = indipendant possibility to belong to each object
        logit = self.soft_aggregation(ps, K, n_objects)    # 1, K, H, W
        # print(logit.shape)    # torch.Size([1, 11, 480, 864])

        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :]

        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]]

        # print(logit.shape)    # torch.Size([1, 11, 480, 854])
        return logit

    def forward(self, frames, masks, n_objects, memorize_every):
        batch_size, n_frames, k, h, w = masks.size()
        est_masks = torch.zeros(batch_size, n_frames, k, h, w).float()
        # Put the tensor to CUDA during training
        if memorize_every == 1:
            est_masks = utils.helpers.var_or_cuda(est_masks)

        keys = None
        values = None
        est_masks[:, 0] = masks[:, 0]
        n_objects = [no.item() for no in n_objects]
        to_memorize = [j for j in range(0, n_frames, memorize_every)]
        for t in range(1, n_frames):
            # Memorize
            prev_mask = utils.helpers.var_or_cuda(est_masks[:, t - 1])
            prev_key, prev_value = self.memorize(frames[:, t - 1], prev_mask, n_objects)
            if t - 1 == 0:
                this_keys, this_values = prev_key, prev_value
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)

            if t - 1 in to_memorize:
                keys, values = this_keys, this_values

            # Segment
            logit = self.segment(frames[:, t], this_keys, this_values, n_objects)
            est_masks[:, t] = F.softmax(logit, dim=1)

        return est_masks
