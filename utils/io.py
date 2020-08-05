# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-02 10:22:03
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-08-05 11:22:57
# @Email:  cshzxie@gmail.com

import io
import numpy as np
import os
import sys

from PIL import Image

# References: http://confluence.sensetime.com/pages/viewpage.action?pageId=44650315
from config import __C as cfg
sys.path.append(cfg.MEMCACHED.LIBRARY_PATH)

mc_client = None
if cfg.MEMCACHED.ENABLED:
    import mc
    mc_client = mc.MemcachedClient.GetInstance(cfg.MEMCACHED.SERVER_CONFIG,
                                               cfg.MEMCACHED.CLIENT_CONFIG)


class IO:
    @classmethod
    def get(cls, file_path):
        if not os.path.exists(file_path):
            return None

        _, file_extension = os.path.splitext(file_path)
        if file_extension in ['.png', '.jpg']:
            return cls._read_img(file_path)
        elif file_extension in ['.flo']:
            return cls._read_flo(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_img(cls, file_path):
        if mc_client is None:
            img = Image.open(file_path)
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector)
            img = Image.open(io.BytesIO(np.frombuffer(buf, np.uint8)))

        return img

    @classmethod
    def _read_flo(cls, file_path):
        if mc_client is None:
            with open(file_path, "rb") as f:
                buf = f.read()
        else:
            pyvector = mc.pyvector()
            mc_client.Get(file_path, pyvector)
            buf = mc.ConvertBuffer(pyvector).tobytes()

        if not buf[:4] == b'PIEH':
            raise Exception('Invalid .flo file format.')

        w = int.from_bytes(buf[4:8], byteorder='little')
        h = int.from_bytes(buf[8:12], byteorder='little')
        flow = np.frombuffer(buf[12:], dtype=np.float32).reshape(h, w, 2)
        return flow
