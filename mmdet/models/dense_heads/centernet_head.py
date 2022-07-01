# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class CenterNetHead(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_keypoint_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_keypoint_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_keypoint_reg=dict(type='L1Loss', loss_weight=1.0),
                 num_keypoints=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(CenterNetHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        # add 
        if num_keypoints is not None:
            self.num_keypoints = num_keypoints
            self.keypoints_heatmap_head = self._build_head(in_channel, feat_channel, num_keypoints)
            self.keypoints_offset_head = self._build_head(in_channel, feat_channel, 2)
            self.keypoints_reg_head = self._build_head(in_channel, feat_channel, 2*num_keypoints)  # 直接回归

            
        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_keypoint_heatmap = build_loss(loss_keypoint_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.loss_keypoint_offset = build_loss(loss_keypoint_offset)
        self.loss_keypoint_reg = build_loss(loss_keypoint_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        for head in [self.wh_head, self.offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()  # 由于pred>0 避免在计算损失时出现Nan
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        keypoint_heatap_pred = self.keypoints_heatmap_head(feat).sigmoid()
        keypoint_offset_pred = self.keypoints_offset_head(feat)
        keypoint_reg_pred = self.keypoints_reg_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred, keypoint_heatap_pred,\
               keypoint_offset_pred, keypoint_reg_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             keypoints_preds=None,
             keypoint_offset_preds=None,
             keypoint_reg_preds=None,
             gt_keypoints=None,
             gt_keypoints_mask=None,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None
            # add keypoints
            keypoints_preds (list[Tensor]): keypoints predicts for all levels with
                shape (B, 2*num_keypoints, H, W).
            gt_keypoints (list[Tensor]): Ground truth keypoints for each image with
                shape (1, 2*num_keypoints) in [x, y] format.
            gt_keypoints_mask (list[Tensor]): Ground truth keypoints for each image 
                witt shape (1, num_keypoints) in [x, y] format.
            

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]
        keypoints_pred = keypoints_preds[0]
        keypoint_offset_pred = keypoint_offset_preds[0]
        keypoint_reg_pred = keypoint_reg_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     gt_keypoints, gt_keypoints_mask,
                                                     center_heatmap_pred.shape,
                                                     img_metas[0]['pad_shape'])

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']
        keypoint_heatmap_target = target_result['keypoint_heatmap_target']
        keypoint_offset_target = target_result['keypoint_offset_target']
        keypoint_offset_target_weight = target_result['keypoint_offset_target_weight']
        keypoint_offset_reg_target = target_result['keypoint_offset_reg_target']
        keypoint_offset_reg_target_weight = target_result['keypoint_offset_reg_target_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)

        loss_keypoint_heatmap = self.loss_keypoint_heatmap(
            keypoints_pred, keypoint_heatmap_target, 
            avg_factor=avg_factor*self.num_keypoints)        
        
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)

        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)

        loss_keypoint_offset = self.loss_keypoint_offset(
            keypoint_offset_pred,
            keypoint_offset_target,
            keypoint_offset_target_weight,
            avg_factor=avg_factor * 2
        )

        loss_keypoint_reg = self.loss_keypoint_reg(
            keypoint_reg_pred,
            keypoint_offset_reg_target,
            keypoint_offset_reg_target_weight,
            avg_factor=avg_factor * 2 * self.num_keypoints
        )

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_keypoint_heatmap=loss_keypoint_heatmap,
            loss_keypoint_offset=loss_keypoint_offset,
            loss_keypoint_reg=loss_keypoint_reg)

    def get_targets(self, gt_bboxes, gt_labels, gt_keypoints,
                    gt_keypoints_mask, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])
        
        keypoint_heatmap_target = gt_keypoints[-1].new_zeros(
            [bs, self.num_keypoints, feat_h, feat_w])
        keypoint_offset_target = gt_keypoints[-1].new_zeros([bs, 2, feat_h, feat_w])
        keypoint_offset_target_weight = gt_keypoints[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        keypoint_offset_reg_target = gt_keypoints[-1].new_zeros([bs,\
            2*self.num_keypoints, feat_h, feat_w])
        keypoint_offset_reg_target_weight = gt_keypoints[-1].new_zeros([bs,\
            2*self.num_keypoints, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            gt_keypoint = gt_keypoints[batch_id]
            gt_keypoint_mask = gt_keypoints_mask[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            # add keypoint part(each object has num_keypoints keypoints)
            # 复用 gaussian_radius
            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

                # add keypoints part
                for k in range(self.num_keypoints):
                    if gt_keypoint_mask[j][k] == 2:
                        keypointx = gt_keypoint[j][k*2] * width_ratio
                        keypointy = gt_keypoint[j][k*2+1] * height_ratio
                        keypointx_int, keypointy_int = keypointx.int(), keypointy.int()
                        
                        gen_gaussian_target(keypoint_heatmap_target[batch_id, k],
                                            [keypointx_int, keypointy_int], radius)

                        keypoint_offset_target[batch_id, 0, keypointy_int, keypointx_int]\
                            = keypointx - keypointx_int
                        keypoint_offset_target[batch_id, 1, keypointy_int, keypointx_int]\
                            = keypointy - keypointy_int
                        
                        keypoint_offset_target_weight[batch_id, :, keypointy_int, keypointx_int] = 1

                        keypoint_offset_reg_target[batch_id, k*2, cty_int, ctx_int] =\
                            keypointx - ctx_int
                        keypoint_offset_reg_target[batch_id, k*2+1, cty_int, ctx_int] =\
                            keypointy - cty_int

                        keypoint_offset_reg_target_weight[batch_id, k*2:k*2+2, cty_int, ctx_int] = 1
                    else:
                        pass


        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight,
            keypoint_heatmap_target=keypoint_heatmap_target,
            keypoint_offset_target=keypoint_offset_target,
            keypoint_offset_target_weight=keypoint_offset_target_weight,
            keypoint_offset_reg_target=keypoint_offset_reg_target,
            keypoint_offset_reg_target_weight=keypoint_offset_reg_target_weight)
        return target_result, avg_factor

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    wh_preds[0][img_id:img_id + 1, ...],
                    offset_preds[0][img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self,
                           center_heatmap_pred,
                           wh_pred,
                           offset_pred,
                           img_meta,
                           rescale=False,
                           with_nms=True):
        """Transform outputs of a single image into bbox results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            wh_pred (Tensor): WH heatmap for current level with shape
                (1, 2, H, W).
            offset_pred (Tensor): Offset for current level with shape
                (1, 2, H, W).
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
                5 represent (tl_x, tl_y, br_x, br_y, score) and the score
                between 0 and 1. The shape of the second tensor in the tuple
                is (n,), and each element represents the class label of the
                corresponding box.
        """
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
                                                                 [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border

        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor'])

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      self.test_cfg)
        return det_bboxes, det_labels

    @force_fp32(apply_to=('keypoint_heatmap_preds', 
                'keypoint_offset_preds', 'keypoint_offset_reg'))
    def get_keypoints  (self,
                        center_heatmap_preds,
                        keypoint_heatmap_preds,
                        keypoint_offset_preds,
                        keypoint_offset_reg,
                        img_metas,
                        rescale=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            keypoint_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_keypoints, H, W).
            keypoint_offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            keypoint_offset_reg (list[Tensor]): Keypoint relative location to 
                center with shape (B, 2*num_keypoint, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (k, num_keypoint, 4) tensor, where 4 
                represent (keypoint_x, keypoint_y, prob_center, prob_keypoint).
                The shape of the second tensor in the tuple is (k,), and
                each element represents the class label of the corresponding
                object.
        """
        assert len(keypoint_heatmap_preds) == \
               len(keypoint_offset_preds) == len(keypoint_offset_reg) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_keypoint_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    keypoint_heatmap_preds[0][img_id:img_id + 1, ...],
                    keypoint_offset_preds[0][img_id:img_id + 1, ...],
                    keypoint_offset_reg[0][img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale))
        return result_list

    def _get_keypoint_single(self,
                            center_heatmap_preds,
                            keypoint_heatmap_preds,
                            keypoint_offset_preds,
                            keypoint_offset_reg,
                            img_meta,
                            rescale=False):
        """Transform outputs of a single image into keypoint results.

        Args:
            center_heatmap_pred (Tensor): Center heatmap for current level with
                shape (1, num_classes, H, W).
            keypoint_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_keypoints, H, W).
            keypoint_offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            keypoint_offset_reg (list[Tensor]): Keypoint relative location to 
                center with shape (B, 2*num_keypoint, H, W)
            img_meta (dict): Meta information of current image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            batch_keypoints (Tensor): Decoded output of keypoint location.keypoints_batch 
                shape in [k, num_keypoint, 4] format.
            batch_labels (Tensor): each element represents the class label of the 
                corresponding object. Shape:[k,]
        """
        batch_det_keypoints, batch_labels = self.decode_keypoint_heatmap(
            center_heatmap_preds,
            keypoint_heatmap_preds,
            keypoint_offset_preds,
            keypoint_offset_reg,
            img_meta['batch_input_shape'],
            kernel=self.test_cfg.local_maximum_kernel)

        batch_labels = batch_labels.view(-1)
        batch_border = batch_det_keypoints[0].new_tensor(img_meta['border'])[...,
                                                                 [2, 0]]
        for batch_det_keypoint in batch_det_keypoints:
            batch_det_keypoint[..., :2] -= batch_border
            if rescale:
                batch_det_keypoint[..., :2] /= batch_det_keypoints[0].new_tensor(
                img_meta['scale_factor'][:2])
        
        return batch_det_keypoints, batch_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

        batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                                 dim=-1)
        return batch_bboxes, batch_topk_labels
    
    def decode_keypoint_heatmap(self,
                                center_heatmap_preds,
                                keypoint_heatmap_preds,
                                keypoint_offset_preds,
                                keypoint_offset_reg,
                                img_shape,
                                k=5,
                                kernel=3):
        """Transform outputs into detections raw keypoint prediction.

        Args:
            center_heatmap_preds (Tensor): center predict heatmap,
               shape (1, num_classes, H, W).
            keypoint_heatmap_preds (Tensor): keypoint predict heatmap,
                shape (1, num_keypoint, H, W).
            keypoint_offset_preds (Tensor): keypoint offset predict, 
                shape (1, 2, H, W).
            keypoint_offset_reg (Tensor): Keypoint relative location to 
                center with shape (1, 2*num_keypoint, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            batch_keypoints (Tensor): Decoded output of keypoint location.keypoints_batch shape in [k, num_keypoint, 4] format.
        """
        height, width = center_heatmap_preds.shape[2:]
        inp_h, inp_w = img_shape
        width_ratio = float(width / inp_w)
        height_ratio = float(height / inp_h)

        center_heatmap_preds = get_local_maximum(
            center_heatmap_preds, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_preds, k=k)  # all num_classes top k
        batch_scores, batch_index, batch_topk_labels = batch_dets
        
        batch_keypoints = []
        for j in range(self.num_keypoints):
            # lj_x lj_y
            offset_reg = transpose_and_gather_feat(
                keypoint_offset_reg[:, 2*j:2*j+2, :, :], batch_index)
            lj_x = topk_xs / width_ratio + offset_reg[..., 0] # keypoint location by reg
            lj_y = topk_ys / height_ratio + offset_reg[..., 1]

            # L_x L_y
            keypoint_heatmap_preds = get_local_maximum(
                keypoint_heatmap_preds, kernel=kernel)
            *batch_dets_tmp, topk_ys_tmp, topk_xs_tmp = get_topk_from_heatmap(
                keypoint_heatmap_preds[:, j:j+1, :, :], k=k)  # single class top k//5
            batch_scores_tmp, batch_index_tmp, batch_topk_labels_tmp = batch_dets_tmp
            
            keypoint_offset = transpose_and_gather_feat(
                keypoint_offset_preds, batch_index_tmp)
            L_x = topk_xs_tmp / width_ratio + keypoint_offset[..., 0] # keypoint location by heatmap
            L_y = topk_ys_tmp / height_ratio + keypoint_offset[..., 1]

            batch_keypoints.append(self._assign_ltoL(lj_x, lj_y, batch_scores,
                L_x, L_y, batch_scores_tmp))
        
        batch_keypoints = torch.stack(batch_keypoints, dim=1)
        return batch_keypoints, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:,
                                                             -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels

    def _assign_ltoL(self,l_xs, l_ys, l_scores, L_xs, L_ys, L_scores):
        """Assign each regressed location (l_x, l_y) to its closest detected 
        keypoint(L_x, L_y). Do certain keypoint.

        Args:
            l_xs, l_ys (Tensor): keypoint reg location predict shape (1, topk1).
            l_scores (Tensor): keypoint reg prob predict shape (1, topk1).
            L_xs, L_ys (Tensor): keypoint heatmap location predict shape (1, topk2).
            L_scores (Tensor): keypoint reg prob predict shape (1, topk2).

        Returns:
            keypoints_batch (Tensor): keypoint location shape (1, topk1, 4).
            4 means [keypoint_x, keypoint_y, prob_center, prob_keypoint].
        """
        bs, topk1 = l_xs.shape
        _, topk2 = L_xs.shape

        assert bs == 1
        l_xs, l_ys = l_xs[0], l_ys[0]
        L_xs, L_ys = L_xs[0], L_ys[0]
        l_score, L_scores = l_scores[0], L_scores[0]
        
        keypoints = []
        for j in range(topk1):
            l_x = l_xs[j]
            l_y = l_ys[j]
            dis_min = float('inf')
            for k in range(topk2):
                dis_tmp = (L_xs[k] - l_x).pow(2) + (L_ys[k] - l_y).pow(2)
                # update
                if dis_tmp < dis_min:
                    dis_min = dis_tmp
                    keypoint_x, keypoint_y = L_xs[k], L_ys[k]
            keypoints.append([keypoint_x, keypoint_y, l_score[j], L_scores[k]])
        
        return torch.Tensor(keypoints) 
