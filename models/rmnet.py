# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-09 11:07:00
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-11-03 23:06:26
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

from extensions.reg_att_map_generator import RegionalAttentionMapGenerator


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
        self.res4 = resnet.layer3    # 1/16, 1024

    def forward(self, in_f):
        x = self.conv1(in_f)
        x = self.bn1(x)
        c1 = self.relu(x)    # 1/2, 64
        x = self.maxpool(c1)    # 1/4, 64
        r2 = self.res2(x)    # 1/4, 256
        r3 = self.res3(r2)    # 1/8, 512
        r4 = self.res4(r3)    # 1/16, 1024
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


class MemoryReader(torch.nn.Module):
    def __init__(self):
        super(MemoryReader, self).__init__()

    def forward(self, m_key, m_val, q_key, q_val):    # m_key: o,c,t,h,w
        B, D_e, T, H, W = m_key.size()
        _, D_o, _, _, _ = m_val.size()

        mi = m_key.view(B, D_e, T * H * W)
        mi = torch.transpose(mi, 1, 2)    # b, THW, emb
        qi = q_key.view(B, D_e, H * W)    # b, emb, HW

        p = torch.bmm(mi, qi)    # b, THW, HW
        p = p / math.sqrt(D_e)
        p = F.softmax(p, dim=1)    # b, THW, HW
        mo = m_val.view(B, D_o, T * H * W)

        mem = torch.bmm(mo, p)    # Weighted-sum B, D_o, HW
        mem = mem.view(B, D_o, H, W)

        mem_val = torch.cat([mem, q_val], dim=1)

        return mem_val, p


class KeyValue(torch.nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.key_conv = torch.nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.value_conv = torch.nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.key_conv(x), self.value_conv(x)


class RMNet(torch.nn.Module):
    def __init__(self, cfg):
        super(RMNet, self).__init__()
        self.cfg = cfg
        self.encoder_memory = EncoderMemory()
        self.encoder_query = EncoderQuery()
        self.kv_memory = KeyValue(1024, keydim=128, valdim=512)
        self.kv_query = KeyValue(1024, keydim=128, valdim=512)
        self.memory = MemoryReader()
        self.decoder = Decoder(256)
        self.att_map_generator = RegionalAttentionMapGenerator()

    def pad_memory(self, mems, n_objects, K):
        pad_mems = []
        B = len(n_objects)

        for mem in mems:
            _, C, H, W = mem.size()
            pad_mem = utils.helpers.var_or_cuda(torch.zeros(B, K, C, 1, H, W), mem.device)
            for i in range(B):
                begin = sum(n_objects[:i])
                end = begin + n_objects[i]
                pad_mem[i, 1:n_objects[i] + 1, :, 0] = mem[begin:end]

            pad_mems.append(pad_mem)

        return pad_mems

    def memorize(self, frame, masks, n_objects):
        # memorize a frame
        B, K, H, W = masks.shape
        # print(frame.shape)    # torch.Size([bs, 3, 480, 910])
        # print(masks.shape)    # torch.Size([bs, n_objects + 1, 480, 910])
        (frame, masks), _ = utils.helpers.pad_divide_by([frame, masks], 16,
                                                          (frame.size()[2], frame.size()[3]))
        # print(frame.shape)    # torch.Size([bs, 3, 480, 912])
        # print(masks.shape)    # torch.Size([bs, n_objects + 1, 480, 912])

        # make batch arg list
        batch_list = {'f': [], 'm': [], 'o': []}
        for i in range(B):
            for o in range(1, n_objects[i] + 1):    # 1 - no
                batch_list['f'].append(frame[i].unsqueeze(0))
                batch_list['m'].append(masks[i, o].unsqueeze(0))
                batch_list['o'].append(
                    (torch.sum(masks[i, 1:o].unsqueeze(0), dim=1) +
                     torch.sum(masks[i, o + 1:n_objects[i] + 1].unsqueeze(0), dim=1)).clamp(0, 1))

        # Make Batch
        for k, v in batch_list.items():
            batch_list[k] = torch.cat(v, dim=0)

        # print(batch_list['f'].shape)  # torch.Size([bs * n_objects, 3, 480, 912])
        # print(batch_list['m'].shape)  # torch.Size([bs * n_objects, 480, 912])
        # print(batch_list['o'].shape)  # torch.Size([bs * n_objects, 480, 912])
        r4, _, _, _, _ = self.encoder_memory(batch_list['f'], batch_list['m'], batch_list['o'])
        # print(r4.shape)       # torch.Size([bs * n_objects, 1024, 30, 57])
        k4, v4 = self.kv_memory(r4)
        # print(k4.shape)       # torch.Size([bs * n_objects, 128, 30, 57])
        # print(v4.shape)       # torch.Size([bs * n_objects, 512, 30, 57])
        k4, v4 = self.pad_memory([k4, v4], n_objects=n_objects, K=K)
        # print(k4.shape)       # torch.Size([bs, n_objects, 128, 1, 30, 57])
        # print(v4.shape)       # torch.Size([bs, n_objects, 512, 1, 30, 57])

        # Generate the regional memory embedding
        att_map, bboxes = self.get_att_map(masks)
        att_map = F.interpolate(att_map, scale_factor=1 / 16).unsqueeze(dim=2).unsqueeze(dim=2)
        # print(att_map.shape)  # torch.Size([bs, n_objects, 1, 1, 30, 57])
        k4 = k4 * att_map
        v4 = v4 * att_map

        return k4, v4, bboxes

    def warp(self, img0, flow):
        # References:
        # - https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py#L139
        B, C, H, W = img0.size()

        x_axis = torch.arange(0, W).view(1, -1).repeat(H, 1)
        y_axis = torch.arange(0, H).view(-1, 1).repeat(1, W)
        x_axis = x_axis.view(1, 1, H, W).repeat(B, 1, 1, 1)
        y_axis = y_axis.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((x_axis, y_axis), 1).float()
        grid = utils.helpers.var_or_cuda(grid, img0.device)
        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        # Fix issue: one of the variables needed for gradient computation has been
        # modified by an inplace operation
        img1 = F.grid_sample(img0.clone(), vgrid, align_corners=True)
        mask = utils.helpers.var_or_cuda(torch.ones(img0.size()), img0.device)
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        img1 = img1 * mask
        return img1, mask

    def get_att_map(self, prev_mask, flow=None):
        if flow is None:
            expt_mask = prev_mask
        else:
            expt_mask, _ = self.warp(prev_mask, flow)

        att_map, bbox = self.att_map_generator(expt_mask)
        return att_map, bbox

    def soft_aggregation(self, ps, K, n_objects):
        B = len(n_objects)
        _, H, W = ps.shape

        em = utils.helpers.var_or_cuda(torch.zeros(B, K, H, W), ps.device)
        for i in range(B):
            begin = sum(n_objects[:i])
            end = begin + n_objects[i]
            em[i, 0] = torch.prod(1 - ps[begin:end], dim=0)    # bg prob
            em[i, 1:n_objects[i] + 1] = ps[begin:end]    # obj prob

        em = torch.clamp(em, 1e-7, 1 - 1e-7)
        logit = torch.log((em / (1 - em)))
        return logit

    def segment(self, frame, att_map, keys, values, prev_bboxes, curr_bbox, n_objects):
        B, K, keydim, T, H, W = keys.shape
        # print(frame.shape)    # torch.Size([bs, 3, 480, 910])
        (frame, att_map), pad = utils.helpers.pad_divide_by([frame, att_map], 16,
                                                            (frame.size()[2], frame.size()[3]))
        # print(frame.shape)    # torch.Size([bs, 3, 480, 912])
        # print(pad)            # (1, 1, 0, 0)
        r4, r3, r2, _, _ = self.encoder_query(frame)
        # print(r4.shape)       # torch.Size([bs, 1024, 30, 57])
        # print(r3.shape)       # torch.Size([bs, 512, 60, 114])
        # print(r2.shape)       # torch.Size([bs, 256, 120, 228])
        k4, v4 = self.kv_query(r4)
        # print(k4.shape)       # torch.Size([bs, 128, 30, 57])
        # print(v4.shape)       # torch.Size([bs, 512, 30, 57])
        batch_list = {
            'k4e': [],
            'v4e': [],
            'r3e': [],
            'r2e': [],
            'key': [],
            'value': [],
            'att_map': []
        }
        for i in range(B):
            _key = keys[i, 1:n_objects[i] + 1]
            _value = values[i, 1:n_objects[i] + 1]
            _att_map = att_map[i, 1:n_objects[i] + 1].unsqueeze(dim=1)
            # expand to ---  no, c, h, w
            _k4e = k4[i].expand(n_objects[i], -1, -1, -1)
            _v4e = v4[i].expand(n_objects[i], -1, -1, -1)
            _r3e = r3[i].expand(n_objects[i], -1, -1, -1)
            _r2e = r2[i].expand(n_objects[i], -1, -1, -1)
            # print(_k4e.shape) # torch.Size([n_objects, 128, 30, 57])
            # print(_v4e.shape) # torch.Size([n_objects, 512, 30, 57])
            # print(_r3e.shape) # torch.Size([n_objects, 512, 60, 114])
            # print(_r2e.shape) # torch.Size([n_objects, 256, 120, 228])
            batch_list['k4e'].append(_k4e)
            batch_list['v4e'].append(_v4e)
            batch_list['r3e'].append(_r3e)
            batch_list['r2e'].append(_r2e)
            batch_list['key'].append(_key)
            batch_list['value'].append(_value)
            batch_list['att_map'].append(_att_map)

        for k, v in batch_list.items():
            batch_list[k] = torch.cat(v, dim=0)

        # memory select kv:(1, K, C, T, H, W)
        # print(keys.shape)     # torch.Size([bs, n_objects, 128, 1, 30, 57])
        # print(values.shape)   # torch.Size([bs, n_objects, 512, 1, 30, 57])

        # Generate the regional query embedding
        att_map = F.interpolate(batch_list['att_map'], scale_factor=1 / 16)
        batch_list['k4e'] = batch_list['k4e'] * att_map
        batch_list['v4e'] = batch_list['v4e'] * att_map

        # Regional Memory Reader
        m4, viz = self.memory(batch_list['key'], batch_list['value'], batch_list['k4e'],
                              batch_list['v4e'])
        # print(m4.shape)       # torch.Size([n_objects, 1024, 30, 57])
        # print(viz.shape)      # torch.Size([n_objects, 3240, 1710])

        logits = self.decoder(m4, batch_list['r3e'], batch_list['r2e'])
        # print(logits.shape)   # torch.Size([n_objects, 2, 480, 912])
        ps = F.softmax(logits, dim=1)    # no, h, w
        # print(ps.shape)       # torch.Size([n_objects, 2, 480, 912])
        ps = ps[:, 1]
        # print(ps.shape)       # torch.Size([n_objects, 480, 912])
        # ps = indipendant possibility to belong to each object
        logit = self.soft_aggregation(ps, K, n_objects)    # 1, K, H, W
        # print(logit.shape)    # torch.Size([bs, n_objects, 480, 912])

        if pad[2] + pad[3] > 0:
            logit = logit[:, :, pad[2]:-pad[3], :]

        if pad[0] + pad[1] > 0:
            logit = logit[:, :, :, pad[0]:-pad[1]]

        # print(logit.shape)    # torch.Size([bs, n_objects, 480, 912])
        return logit

    def forward(self, frames, masks, optical_flows, n_objects, memorize_every, device=None):
        batch_size, n_frames, _, h, w = frames.size()
        k = masks.size(2)
        est_masks = torch.zeros(batch_size, n_frames, k, h, w).float()
        # Fix Assertion Error:  all(map(lambda i: i.is_cuda, inputs))
        # The value of device is set in utils/eval_server.py for full set evaluation
        if torch.cuda.device_count() > 1 and device is None:
            est_masks = utils.helpers.var_or_cuda(est_masks)

        keys = None
        values = None
        bboxes = None
        est_masks[:, 0] = masks[:, 0]
        n_max_objects = [torch.max(no).item() for no in n_objects]
        existing_objects = [
            torch.unique(torch.argmax(masks[i, 0], dim=0)).cpu().tolist()
            for i in range(batch_size)
        ]

        # Set the frames to memorize
        to_memorize = [j for j in range(0, n_frames, memorize_every)]
        contains_new_objects = [
            j for j in range(1, n_frames) if (n_objects[:, j] != n_objects[:, j - 1]).any()
        ]

        for t in range(1, n_frames):
            # Memorize
            prev_mask = utils.helpers.var_or_cuda(est_masks[:, t - 1], device)
            prev_frame = utils.helpers.var_or_cuda(frames[:, t - 1], device)
            prev_key, prev_value, prev_bbox = self.memorize(prev_frame, prev_mask, n_max_objects)

            if t - 1 == 0:
                this_keys, this_values = prev_key, prev_value
                this_bboxes = prev_bbox.unsqueeze(dim=2)
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)
                this_bboxes = torch.cat([bboxes, prev_bbox.unsqueeze(dim=2)], dim=2)

            if t - 1 in to_memorize or t - 1 in contains_new_objects:
                keys, values = this_keys, this_values
                bboxes = this_bboxes

            # Segment
            curr_frame = utils.helpers.var_or_cuda(frames[:, t], device)
            curr_flow = utils.helpers.var_or_cuda(optical_flows[:, t], device)
            reg_att_map, curr_bbox = self.get_att_map(prev_mask, curr_flow)
            logit = self.segment(curr_frame, reg_att_map, this_keys, this_values, this_bboxes,
                                 curr_bbox, n_max_objects)

            # Detect new objects
            if t in contains_new_objects:
                for i in range(batch_size):
                    for j in torch.unique(torch.argmax(masks[i, t], dim=0)).cpu().tolist():
                        if j not in existing_objects[i]:
                            existing_objects[i].append(j)
                            # torch.min(logit) = -16.1181, torch.max(logit) = 15.9424
                            logit[i, j] = masks[i, t, j].float() * 32.0605 - 16.1181

            # Set the prob. of non-existing objects to zeros
            for i in range(batch_size):
                for j in range(n_max_objects[i] + 1):
                    if j not in existing_objects[i]:
                        logit[i, j] = -16.1181

            est_masks[:, t] = F.softmax(logit, dim=1)

        return est_masks
