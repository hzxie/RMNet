# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-04-12 13:44:33
# @Email:  cshzxie@gmail.com
#
# Maintainers
# - David Martin <dmartin@eecs.berkeley.edu>
# - Yunfeng Zhang <zhangyunfeng@sensetime.com>
# - Anni Xu <xuanni@sensetime.com>
# - Haozhe Xie <cshzxie@gmail.com>

import logging
import numpy as np
import skimage.morphology

import utils.helpers


class Metrics(object):
    ITEMS = [{
        'name': 'J-Mean',
        'enabled': True,
        'eval_func': 'cls._get_j_mean',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'F-Mean',
        'enabled': True,
        'eval_func': 'cls._get_f_mean',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'JF-Mean',
        'enabled': True,
        'eval_func': 'cls._get_jf_mean',
        'is_greater_better': True,
        'init_value': 0
    }]

    @classmethod
    def get(cls, pred, gt):
        _items = cls.items()
        _values = [0] * len(_items)
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()

        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            if item['name'] == 'JF-Mean':
                # _values[0] -> J-Mean, _values[1] -> F-Mean
                _values[i] = eval_func(_values[0], _values[1])
            else:
                _values[i] = eval_func(pred, gt)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_j_mean(cls, pred, gt):
        n_frames, _, _ = gt.shape
        jaccard = []
        for i in range(n_frames):
            n_objects = np.max(gt)
            _gt = utils.helpers.to_onehot(gt[i], n_objects + 1)
            _pred = utils.helpers.to_onehot(pred[i], n_objects + 1)

            for j in range(1, n_objects + 1):
                jaccard.append(cls._get_iou(_pred[j], _gt[j]))

        return np.nanmean(jaccard)

    @classmethod
    def _get_iou(cls, segmentation, annotation):
        """
        Compute region similarity as the Jaccard Index.

        Arguments:
            segmentation (ndarray): binary segmentation map.
            annotation   (ndarray): binary annotation   map.

        Return:
            jaccard (float): region similarity
        """
        segmentation = segmentation.astype(np.bool)
        annotation = annotation.astype(np.bool)

        if np.isclose(np.sum(annotation), 0) and np.isclose(np.sum(segmentation), 0):
            return 1
        else:
            return np.sum((annotation & segmentation)) / \
                   np.sum((annotation | segmentation), dtype=np.float32)

    @classmethod
    def _get_f_mean(cls, pred, gt):
        n_frames, _, _ = gt.shape
        f_score = []
        for i in range(n_frames):
            n_objects = np.max(gt)
            _gt = utils.helpers.to_onehot(gt[i], n_objects + 1)
            _pred = utils.helpers.to_onehot(pred[i], n_objects + 1)

            for j in range(1, n_objects + 1):
                f_score.append(cls._get_f_score(_pred[j], _gt[j]))

        return np.nanmean(f_score)

    @classmethod
    def _get_f_score(cls, foreground_mask, gt_mask, bound_th=0.008):
        """
        Compute mean,recall and decay from per-frame evaluation.
        Calculates precision/recall for boundaries between foreground_mask and
        gt_mask using morphological operators to speed it up.

        Arguments:
            foreground_mask (ndarray): binary segmentation image.
            gt_mask         (ndarray): binary annotated image.

        Returns:
            f_score (float): boundaries F-measure
        """
        assert np.atleast_3d(foreground_mask).shape[2] == 1

        bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

        # Get the pixel boundaries of both masks
        fg_boundary = cls.seg2bmap(foreground_mask)
        gt_boundary = cls.seg2bmap(gt_mask)

        fg_dil = skimage.morphology.binary_dilation(fg_boundary,
                                                    skimage.morphology.disk(bound_pix))
        gt_dil = skimage.morphology.binary_dilation(gt_boundary,
                                                    skimage.morphology.disk(bound_pix))

        # Get the intersection
        gt_match = gt_boundary * fg_dil
        fg_match = fg_boundary * gt_dil

        # Area of the intersection
        n_fg = np.sum(fg_boundary)
        n_gt = np.sum(gt_boundary)

        # % Compute precision and recall
        if n_fg == 0 and n_gt > 0:
            precision = 1
            recall = 0
        elif n_fg > 0 and n_gt == 0:
            precision = 0
            recall = 1
        elif n_fg == 0 and n_gt == 0:
            precision = 1
            recall = 1
        else:
            precision = np.sum(fg_match) / float(n_fg)
            recall = np.sum(gt_match) / float(n_gt)

        # Compute F measure
        return 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    @classmethod
    def seg2bmap(cls, seg, width=None, height=None):
        """
        From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries.  The boundary pixels are offset by 1/2 pixel towards the
        origin from the actual segment boundary.

        Arguments:
            seg     : Segments labeled from 1..k.
            width     : Width of desired bmap  <= seg.shape[1]
            height  :   Height of desired bmap <= seg.shape[0]

        Returns:
            bmap (ndarray): Binary boundary map.
        """
        seg = seg.astype(np.bool)
        seg[seg > 0] = 1

        assert np.atleast_3d(seg).shape[2] == 1

        width = seg.shape[1] if width is None else width
        height = seg.shape[0] if height is None else height

        h, w = seg.shape[:2]

        ar1 = float(width) / float(height)
        ar2 = float(w) / float(h)

        assert not (width > w | height > h | abs(ar1 - ar2) > 0.01), \
            'Can''t convert %dx%d seg to %dx%d bmap.' % (w, h, width, height)

        e = np.zeros_like(seg)
        s = np.zeros_like(seg)
        se = np.zeros_like(seg)

        e[:, :-1] = seg[:, 1:]
        s[:-1, :] = seg[1:, :]
        se[:-1, :-1] = seg[1:, 1:]

        b = seg ^ e | seg ^ s | seg ^ se
        b[-1, :] = seg[-1, :] ^ e[-1, :]
        b[:, -1] = seg[:, -1] ^ s[:, -1]
        b[-1, -1] = 0

        if w == width and h == height:
            bmap = b
        else:
            bmap = np.zeros((height, width))
            for x in range(w):
                for y in range(h):
                    if b[y, x]:
                        j = 1 + np.floor((y - 1) + height / h)
                        i = 1 + np.floor((x - 1) + width / h)
                        bmap[j, i] = 1

        return bmap

    @classmethod
    def _get_jf_mean(cls, j_mean, f_mean):
        return (j_mean + f_mean) / 2.

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
