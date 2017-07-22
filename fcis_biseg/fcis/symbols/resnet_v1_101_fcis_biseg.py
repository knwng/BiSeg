# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Haozhi Qi, Guodong Zhang, Yi Li
# --------------------------------------------------------

import cPickle
import mxnet as mx
import numpy as np
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_annotator import *
from operator_py.box_parser import *
from operator_py.box_annotator_ohem import *
import symbol_fcnxs
# from net_util import get_resnet_v1_conv4, get_resnet_v1_conv5, get_rpn
from net_util import *


class resnet_v1_101_fcis_biseg(Symbol):

    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3) # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_symbol(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS

        # input init
        if is_train:
            data = mx.sym.Variable(name='data')
	    # data_ss = mx.sym.Variable(name='data_ss')
            im_info = mx.sym.Variable(name='im_info')
            gt_boxes = mx.sym.Variable(name='gt_boxes')
            gt_masks = mx.sym.Variable(name='gt_masks')
            ss_masks = mx.sym.Variable(name='ss_masks')
            rpn_label = mx.sym.Variable(name='proposal_label')
            rpn_bbox_target = mx.sym.Variable(name='proposal_bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='proposal_bbox_weight')
        else:
            data = mx.sym.Variable(name="data")
	    # data_ss = mx.sym.Variable(name="data_ss")
            im_info = mx.sym.Variable(name="im_info")
	
        # shared convolutional layers
        # conv_feat = get_resnet_v1_conv4(data)
        # res4b22_relu, res3b3_relu, res2c_relu = get_resnet_v1_conv4(data, eps=self.eps)
        conv_feat, res3_relu, _ = get_resnet_v1_conv4(data, eps=self.eps)
        # res5
        # relu1 = get_resnet_v1_conv5(conv_feat)
        relu1 = get_resnet_v1_conv5(conv_feat, eps=self.eps)
	'''
        # semantic segmentation using fcn-8s
	if is_train:
       		fcnx_1 = symbol_fcnxs.get_fcn32s_symbol(data=relu1, data_ori=data,  
			masks=ss_masks, numclass=num_classes, workspace_default=3072) # 1536
	else:
		fcnx_1 = symbol_fcnxs.get_fcn32s_symbol(data=relu1, data_ori=data, numclass=num_classes, workspace_default=3072)
	fcnx = fcnx_1
	'''

        # Add RPN from conv4
        rpn_cls_score, rpn_bbox_pred = get_rpn(conv_feat, num_anchors)
        # rpn_bbox_pred indicate bbox parameters

        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            # RPN bbox loss (smooth l1) for rectangle parameters estimation
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / cfg.TRAIN.RPN_BATCH_SIZE)

            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            rpn_cls_act_reshape = mx.sym.Reshape(
                data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            # result of 2-categories classifier
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    nms_threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
            group = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape, gt_masks=gt_masks,
                                  op_type='proposal_annotator',
                                  num_classes=num_reg_classes, mask_size=cfg.MASK_SIZE, binary_thresh=cfg.TRAIN.BINARY_THRESH,
                                  batch_images=cfg.TRAIN.BATCH_IMAGES, cfg=cPickle.dumps(cfg),
                                  batch_rois=cfg.TRAIN.BATCH_ROIS, fg_fraction=cfg.TRAIN.FG_FRACTION)
            rois = group[0]
            label = group[1]
            bbox_target = group[2]
            bbox_weight = group[3]
            mask_reg_targets = group[4]
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(
                data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(
                data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(
                data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.Proposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    nms_threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)


        # conv new 1
        # relu1 is conv5(res5) of resnet-101
        # apply 2048->1024 transform
        if cfg.TRAIN.CONVNEW3:
            conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name='conv_new_1', attr={'lr_mult':'3.00'})
        else:
            conv_new_1 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=1024, name='conv_new_1')
        relu_new_1 = mx.sym.Activation(data=conv_new_1, act_type='relu', name='relu_new_1')
	'''
        # semantic segmentation using fcn-8s
	if is_train:
       		fcnx_1 = symbol_fcnxs.get_fcn16s_symbol(data=relu_new_1, res3=res3_relu, data_ori=data,  
			masks=ss_masks, numclass=num_classes, workspace_default=3072) # 1536
	else:
		fcnx_1 = symbol_fcnxs.get_fcn16s_symbol(data=relu_new_1, res3=res3_relu, data_ori=data, 
			numclass=num_classes, workspace_default=3072)
	fcnx = fcnx_1[0]
	fcnx_1 = fcnx_1[1]
	'''
	#
	#
	#
	lateral_conn_0 = mx.sym.Convolution(data=relu1, kernel=(1, 1), num_filter=num_classes, name='lateral_connection')
	lateral_conn = mx.sym.sigmoid(data=lateral_conn_0, name='lateral_sigmoid')
	fcnx = mx.sym.ROIPooling(data=lateral_conn, rois=rois, pooled_size=(21, 21), spatial_scale=0.0625, name='roipool_fcn')
	if is_train:
		ss_masks_resize = mx.sym.Pooling(data=ss_masks, kernel=(16, 16), pool_type='max', stride=(16, 16), name='ss_masks_pool')
		# lateral_resize = mx.sym.Deconvolution(data=lateral_conn_0, kernel=(2, 2), stride=(2, 2), num_filter=num_classes, adj=(1,1), name="lateral_resize")
		lateral_resize = mx.sym.Deconvolution(data=lateral_conn_0, kernel=(2, 2), stride=(1, 1), num_filter=num_classes, name="lateral_resize")
		lateral_crop = mx.sym.Crop(*[lateral_resize, ss_masks_resize], center_crop=True, name="lateral_crop")
		# ss_pred = mx.sym.SoftmaxOutput(data=lateral_crop, label=ss_masks_resize, multi_output=True, use_ignore=True, ignore_label=-1, name="fcn_softmax", normalization='valid', grad_scale=1./num_classes)
		ss_pred = mx.sym.SoftmaxOutput(data=lateral_crop, label=ss_masks_resize, multi_output=True, use_ignore=True, ignore_label=-1, name="fcn_softmax", normalization='valid', grad_scale=1e-8)
		# ss_pred = mx.sym.SoftmaxOutput(data=lateral_crop, label=ss_masks_resize, name="fcn_softmax", normalization='valid', grad_scale=1e-5)
		# ss_pred = mx.sym.SoftmaxOutput(data=lateral_crop, label=ss_masks_resize, use_ignore=True, ignore_label=-1, name="fcn_softmax", normalization='valid')
		# ss_pred = mx.sym.SoftmaxOutput(data=lateral_crop, label=ss_masks_resize, multi_output=True, use_ignore=True, ignore_label=255, name="fcn_softmax", normalization='valid', grad_scale=1./num_classes)
	#
	#
	#
        # 7 represent the size of regular grid
        fcis_cls_seg = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*num_classes*2,
                                          name='fcis_cls_seg')
        fcis_bbox = mx.sym.Convolution(data=relu_new_1, kernel=(1, 1), num_filter=7*7*4*num_reg_classes,
                                       name='fcis_bbox')
        psroipool_cls_seg_0 = mx.contrib.sym.PSROIPooling(name='psroipool_cls_seg_0', data=fcis_cls_seg, rois=rois,
                                                        group_size=7, pooled_size=21, output_dim=num_classes*2, spatial_scale=0.0625)
        cls_seg_split = mx.sym.split(name='cls_seg_split', data=psroipool_cls_seg_0, axis=1, num_outputs=2)
        posterior = fcnx * cls_seg_split[0]
        psroipool_cls_seg = mx.sym.concat(posterior, cls_seg_split[1], dim=1, name='psroipool_cls_seg')
        psroipool_bbox_pred = mx.contrib.sym.PSROIPooling(name='psroipool_bbox', data=fcis_bbox, rois=rois,
                                                          group_size=7, pooled_size=21,  output_dim=num_reg_classes*4, spatial_scale=0.0625)
        if is_train:
            # classification path
            # input psroipool_cls_seg
            psroipool_cls = mx.contrib.sym.ChannelOperator(name='psroipool_cls', data=psroipool_cls_seg, group=num_classes, op_type='Group_Max')
            cls_score = mx.sym.Pooling(name='cls_score', data=psroipool_cls, pool_type='avg', global_pool=True, kernel=(21, 21))
            cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
            # mask regression path
            # input label
            label_seg = mx.sym.Reshape(name='label_seg', data=label, shape=(-1, 1, 1, 1))
            seg_pred = mx.contrib.sym.ChannelOperator(name='seg_pred', data=psroipool_cls_seg, pick_idx=label_seg, group=num_classes, op_type='Group_Pick', pick_type='Label_Pick')
            # bbox regression path
            bbox_pred = mx.sym.Pooling(name='bbox_pred', data=psroipool_bbox_pred, pool_type='avg', global_pool=True, kernel=(21, 21))
            bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))
        else:
            # classification path
            psroipool_cls = mx.contrib.sym.ChannelOperator(name='psroipool_cls', data=psroipool_cls_seg, group=num_classes, op_type='Group_Max')
            cls_score = mx.sym.Pooling(name='cls_score', data=psroipool_cls, pool_type='avg', global_pool=True,
                                       kernel=(21, 21))
            cls_score = mx.sym.Reshape(name='cls_score_reshape', data=cls_score, shape=(-1, num_classes))
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            # mask regression path
            score_seg = mx.sym.Reshape(name='score_seg', data=cls_prob, shape=(-1, num_classes, 1, 1))
            seg_softmax = mx.contrib.sym.ChannelOperator(name='seg_softmax', data=psroipool_cls_seg, group=num_classes, op_type='Group_Softmax')
            # use softmax to differentiate segment+ & segment-
            seg_pred = mx.contrib.sym.ChannelOperator(name='seg_pred', data=seg_softmax, pick_idx=score_seg, group=num_classes, op_type='Group_Pick', pick_type='Score_Pick')
            # bbox regression path
            bbox_pred = mx.sym.Pooling(name='bbox_pred', data=psroipool_bbox_pred, pool_type='avg', global_pool=True,
                                       kernel=(21, 21))
            bbox_pred = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred, shape=(-1, 4 * num_reg_classes))

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, mask_targets_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                num_reg_classes=num_reg_classes, roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM, cfg=cPickle.dumps(cfg),
                                                cls_score=cls_score, seg_pred=seg_pred, bbox_pred=bbox_pred, labels=label,
                                                mask_targets=mask_reg_targets, bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem, normalization='valid',
                                                use_ignore=True, ignore_label=-1, grad_scale=cfg.TRAIN.LOSS_WEIGHT[0])
                seg_prob = mx.sym.SoftmaxOutput(name='seg_prob', data=seg_pred, label=mask_targets_ohem, multi_output=True,
                                                normalization='null', use_ignore=True, ignore_label=-1,
                                                grad_scale=cfg.TRAIN.LOSS_WEIGHT[1] / cfg.TRAIN.BATCH_ROIS_OHEM)
                bbox_loss_t = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_t', scalar=1.0, data=(bbox_pred - bbox_target))
                # bbox_loss_t = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_t', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_t, grad_scale=cfg.TRAIN.LOSS_WEIGHT[2] / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid',
                                                use_ignore=True, ignore_label=-1, grad_scale=cfg.TRAIN.LOSS_WEIGHT[0])
                seg_prob = mx.sym.SoftmaxOutput(name='seg_prob', data=seg_pred, label=mask_reg_targets, multi_output=True,
                                                normalization='null', use_ignore=True, ignore_label=-1,
                                                grad_scale=cfg.TRAIN.LOSS_WEIGHT[1] / cfg.TRAIN.BATCH_ROIS)
                # bbox_loss_t = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_t', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss_t = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_t', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_t, grad_scale=cfg.TRAIN.LOSS_WEIGHT[2] / cfg.TRAIN.BATCH_ROIS)
                rcnn_label = label

            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')
            # group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, seg_prob, mx.sym.BlockGrad(mask_reg_targets), mx.sym.BlockGrad(rcnn_label)])
            # group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, seg_prob, mx.sym.BlockGrad(mask_reg_targets), mx.sym.BlockGrad(rcnn_label), fcnx_1, mx.sym.BlockGrad(ss_masks)])
            # group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, seg_prob, mx.sym.BlockGrad(mask_reg_targets), mx.sym.BlockGrad(rcnn_label), ss_pred, mx.sym.BlockGrad(ss_masks_crop)])
            group = mx.sym.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, seg_prob, mx.sym.BlockGrad(mask_reg_targets), mx.sym.BlockGrad(rcnn_label), ss_pred, mx.sym.BlockGrad(ss_masks_resize)])
        else:
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            if cfg.TEST.ITER == 2:
                rois_iter2 = mx.sym.Custom(bottom_rois=rois, bbox_delta=bbox_pred, im_info=im_info, cls_prob=cls_prob,
                                           name='rois_iter2', b_clip_boxes=True, bbox_class_agnostic=True,
                                           bbox_means=tuple(cfg.TRAIN.BBOX_MEANS), bbox_stds=tuple(cfg.TRAIN.BBOX_STDS), op_type='BoxParser')
                # rois = mx.sym.Concat(*[rois, rois_iter2], dim=0, name='rois')
                # perform bayesian inference on inside score maps
                psroipool_cls_seg_iter2_0 = mx.contrib.sym.PSROIPooling(name='psroipool_cls_seg_0', data=fcis_cls_seg, rois=rois_iter2,
                                                                group_size=7, pooled_size=21,
                                                                output_dim=num_classes*2, spatial_scale=0.0625)
                cls_seg_split_iter2 = mx.sym.split(name='cls_seg_split', data=psroipool_cls_seg_iter2_0, axis=1, num_outputs=2)
                posterior_iter2 = fcnx * cls_seg_split_iter2[0]
                psroipool_cls_seg_iter2 = mx.sym.concat(posterior_iter2, cls_seg_split_iter2[1], dim=1, name='psroipool_cls_seg') 
                psroipool_bbox_pred_iter2 = mx.contrib.sym.PSROIPooling(name='psroipool_bbox', data=fcis_bbox, rois=rois_iter2,
                                                                group_size=7, pooled_size=21,
                                                                output_dim=num_reg_classes*4, spatial_scale=0.0625)

                # classification path
                psroipool_cls_iter2 = mx.contrib.sym.ChannelOperator(name='psroipool_cls', data=psroipool_cls_seg_iter2, group=num_classes,
                                                             op_type='Group_Max')
                cls_score_iter2 = mx.sym.Pooling(name='cls_score', data=psroipool_cls_iter2, pool_type='avg', global_pool=True, kernel=(21, 21), stride=(21,21))

                cls_score_iter2 = mx.sym.Reshape(name='cls_score_reshape', data=cls_score_iter2, shape=(-1, num_classes))
                cls_prob_iter2 = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score_iter2)
                # mask regression path
                score_seg_iter2 = mx.sym.Reshape(name='score_seg', data=cls_prob_iter2, shape=(-1, num_classes, 1, 1))
                seg_softmax_iter2 = mx.contrib.sym.ChannelOperator(name='seg_softmax', data=psroipool_cls_seg_iter2, group=num_classes, op_type='Group_Softmax')
                seg_pred_iter2 = mx.contrib.sym.ChannelOperator(name='seg_pred', data=seg_softmax_iter2, pick_idx=score_seg_iter2, group=num_classes, op_type='Group_Pick', pick_type='Score_Pick')

                # bbox regression path
                bbox_pred_iter2 = mx.sym.Pooling(name='bbox_pred', data=psroipool_bbox_pred_iter2, pool_type='avg', global_pool=True, kernel=(21, 21), stride=(21,21))
                bbox_pred_iter2 = mx.sym.Reshape(name='bbox_pred_reshape', data=bbox_pred_iter2, shape=(-1, 4 * num_reg_classes))

                rois = mx.sym.Concat(*[rois, rois_iter2], dim=0, name='rois')
                cls_prob = mx.sym.Concat(*[cls_prob, cls_prob_iter2], dim=0, name='cls_prob')
                seg_pred = mx.sym.Concat(*[seg_pred, seg_pred_iter2], dim=0, name='seg_pred')
                bbox_pred = mx.sym.Concat(*[bbox_pred, bbox_pred_iter2], dim=0, name='box_pred')
            # reshape output
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, cls_prob, bbox_pred, seg_pred])

        self.sym = group
        return group

    def init_weight(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['fcis_cls_seg_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fcis_cls_seg_weight'])
        arg_params['fcis_cls_seg_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fcis_cls_seg_bias'])
        arg_params['fcis_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fcis_bbox_weight'])
        arg_params['fcis_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fcis_bbox_bias'])
        arg_params['lateral_connection_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['lateral_connection_weight'])
        arg_params['lateral_connection_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['lateral_connection_bias'])
        arg_params['lateral_resize_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['lateral_resize_weight'])
	'''
        # init fcn-32s parameters
        arg_params['bigscore_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bigscore_weight'])
        arg_params['score2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score2_weight'])
        arg_params['score_pool4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score_pool4_weight'])
        arg_params['score_pool4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['score_pool4_bias'])
        arg_params['score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score_weight'])
        arg_params['score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['score_bias'])
        arg_params['score_pool3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score_pool3_weight'])
        arg_params['score_pool3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['score_pool3_bias'])
        arg_params['score4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score4_weight'])
	'''
