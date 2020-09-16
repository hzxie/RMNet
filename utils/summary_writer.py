# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2020-04-19 12:52:36
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-09-16 12:52:03
# @Email:  cshzxie@gmail.com

import logging
import os
import tensorboardX
try:
    import pavi
except Exception as ex:
    logging.warning(ex)


class SummaryWriter(object):
    def __init__(self, cfg, phase):
        if cfg.PAVI.ENABLED:
            task = '%s/%s' % (cfg.CONST.EXP_NAME, phase)
            project = cfg.PAVI.PROJECT_NAME

            self.writer = pavi.SummaryWriter(task=task, labels=phase, project=project)
        else:
            self.writer = tensorboardX.SummaryWriter(os.path.join(cfg.DIR.LOGS, phase))

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag, img_tensor, global_step=None):
        self.writer.add_image(tag, img_tensor, global_step)

    def close(self):
        if type(self.writer) == tensorboardX.writer.SummaryWriter:
            self.writer.close()
