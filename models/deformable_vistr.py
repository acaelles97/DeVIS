# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ClassEmbdInstanceAttn(nn.Module):

    def __init__(self, hidden_dim, n_heads, num_classes, num_frames):
        super().__init__()
        # self attention trajectory
        self.n_frames = num_frames
        self.self_attn_inst = nn.MultiheadAttention(hidden_dim, n_heads, dropout=0.1)
        self.dropout2_inst = nn.Dropout(0.1)
        self.norm2_inst = nn.LayerNorm(hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

    def forward(self, tgt, query_pos):
        embd_per_frame = tgt.shape[1] // self.n_frames
        tgt = tgt.reshape([self.n_frames, embd_per_frame, tgt.shape[-1]])
        query_pos = query_pos.reshape([self.n_frames, embd_per_frame, tgt.shape[-1]])
        q = k = tgt + query_pos
        tgt2 = self.self_attn_inst(q, k, tgt)[0]
        tgt = tgt + self.dropout2_inst(tgt2)
        tgt = self.norm2_inst(tgt)
        tgt = tgt.flatten(0, 1)[None]

        return self.class_embed(tgt)


class DeformableVisTR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_frames, num_queries, num_feature_levels, aux_loss=True, with_box_refine=False,
                 two_stage=False, use_trajectory_queries=False, with_ref_point_refine=False, with_gradient=False, with_single_class_embed=False, with_class_inst_attn=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.backbone = backbone
        self.num_queries = num_queries
        self.use_trajectory_queries = use_trajectory_queries
        self.with_ref_point_refine = with_ref_point_refine
        self.with_gradient = with_gradient
        self.with_class_inst_attn = with_class_inst_attn
        self.transformer = transformer
        hidden_dim = transformer.d_model

        if with_class_inst_attn:
            self.class_embed = ClassEmbdInstanceAttn(hidden_dim, transformer.nhead_dec, num_classes, num_frames)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.num_frames = num_frames
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if self.num_feature_levels == 1:
            num_channels = [self.backbone.num_channels[3]]

        else:
            num_channels = self.backbone.num_channels[-3:]

        if num_feature_levels > 1:
            input_proj_list = []
            num_backbone_outs = len(self.backbone.strides) - 1

            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)

        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.with_class_inst_attn:
            self.class_embed.class_embed.bias.data = torch.ones(num_classes + 1) * bias_value
        else:
            self.class_embed.bias.data = torch.ones(num_classes + 1) * bias_value

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            assert not with_ref_point_refine
            if with_single_class_embed:
                self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            else:
                self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
            if with_ref_point_refine:
                ref_point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
                nn.init.constant_(ref_point_embed.layers[-1].weight.data, 0)
                nn.init.constant_(ref_point_embed.layers[-1].bias.data, 0)
                self.transformer.decoder.ref_point_embed = _get_clones(ref_point_embed, num_pred)

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        else:
            if not use_trajectory_queries:
                num_trajectories = num_queries // num_frames
                new_weights = nn.init.normal(torch.empty((num_trajectories, 512))).repeat(num_frames, 1, 1)
                with torch.no_grad():
                    self.query_embed.weight =  nn.Parameter(new_weights.flatten(0, 1))

    def forward(self, samples: NestedTensor, targets: dict = None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        features_all = features
        if self.num_feature_levels == 1:
            features, pos = [features[-1]], [pos[-1]]

        else:
            features, pos = features[1:], pos[1:]
        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src_proj = self.input_proj[l](src)
            srcs.append(src_proj)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src_proj = self.input_proj[l](features[-1].tensors)
                else:
                    src_proj = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src_proj.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src_proj, mask)).to(src_proj.dtype)
                srcs.append(src_proj)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight

        hs, query_pos, memory, init_reference, inter_references, level_start_index, valid_ratios, spatial_shapes, targets  = self.transformer(srcs, masks, pos, query_embeds, targets)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if self.with_class_inst_attn:
                outputs_class = self.class_embed[lvl](hs[lvl], query_pos)
            else:
                outputs_class = self.class_embed[lvl](hs[lvl])

            outputs_classes.append(outputs_class)

            if self.with_gradient:
                outputs_coord = inter_references[lvl]
            else:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
                tmp = self.bbox_embed[lvl](hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()

            outputs_coords.append(outputs_coord)


        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        offset = 0
        memory_slices = []
        for src in srcs:
            n, c, h, w = src.shape
            memory_slice = memory[:, offset:offset + h * w * self.num_frames].permute(0, 2, 1).reshape(n // self.num_frames, c, self.num_frames, h, w)
            memory_slices.append(memory_slice)
            offset += h * w * self.num_frames

        return out, features_all, memory, memory_slices, hs, query_pos, srcs, masks, init_reference, inter_references, level_start_index, valid_ratios, spatial_shapes, targets

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class AllPostProcessor(nn.Module):
    def __init__(self, focal_loss, num_frames, top_k_inference, use_instance_level_classes):
        super().__init__()
        self.focal_loss = focal_loss
        self.num_frames = num_frames
        self.top_k_inference = top_k_inference
        self.use_instance_level_classes = use_instance_level_classes

    # def top_k_process_results_(self, output_prob, boxes):
    #
    #     scores, top_k_indexes = torch.topk(output_prob.view(output_prob.shape[0], -1), self.top_k_predictions, dim=1)
    #     query_top_k_indexes = top_k_indexes //  output_prob.shape[2]
    #     labels = top_k_indexes % output_prob.shape[2]
    #     boxes = torch.gather(boxes, 1, query_top_k_indexes.unsqueeze(-1).repeat(1, 1, 4))
    #     return scores, labels, boxes, query_top_k_indexes
    #
    # def top_k_process_results(self, output_prob, boxes):
    #     top_score_per_embeding, label_per_embedding = output_prob.max(-1)
    #     scores, query_top_k_indexes = torch.topk(top_score_per_embeding, self.top_k_predictions, dim=1)
    #     labels = torch.gather(label_per_embedding, 1, query_top_k_indexes)
    #     boxes = torch.gather(boxes, 1, query_top_k_indexes.unsqueeze(-1).repeat(1, 1, 4))
    #     return scores, labels, boxes, query_top_k_indexes

    def process_boxes(self, boxes, tgt_size):
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = torch.tensor([tgt_size]).unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes.cpu() * scale_fct[:, None, :]
        boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]], 0, img_w.item())
        boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]], 0, img_h.item())
        boxes = boxes.reshape(self.num_frames, -1, boxes.shape[-1]).detach()
        return boxes

    @torch.no_grad()
    def forward(self, outputs, tgt_size, video_length, process_boxes=True):
        # end of model inference
        if self.focal_loss:
            logits = outputs['pred_logits'].sigmoid()
        else:
            logits = outputs['pred_logits'].softmax(-1)[0, :, :-1]

        num_trajectories = outputs['pred_boxes'].shape[1] // self.num_frames
        assert self.focal_loss
        if self.use_instance_level_classes:
            traj_probs = logits[0].flatten()
        else:
            pred_logits = logits.reshape(self.num_frames, num_trajectories, logits.shape[-1])
            traj_probs = pred_logits[:video_length].transpose(0, 1).mean(1).flatten()

        _, top_k_indexes = torch.topk(traj_probs, self.top_k_inference, dim=0)
        query_top_k_indexes = top_k_indexes // logits.shape[2]

        labels = top_k_indexes % logits.shape[2]
        pred_classes =  labels.repeat(video_length, 1) + 1
        if self.use_instance_level_classes:
            pred_scores = logits[0, query_top_k_indexes, labels]
            pred_scores = pred_scores.repeat(video_length, 1)
        else:
            pred_scores = pred_logits[:, query_top_k_indexes, labels]

        results = {
            "scores": pred_scores[:video_length],
            "labels": pred_classes[:video_length],
        }

        if process_boxes:
            boxes = outputs['pred_boxes'][0]
            pred_boxes = self.process_boxes(boxes, tgt_size)
            pred_boxes = pred_boxes[:, query_top_k_indexes]
            results[ "boxes"]= pred_boxes[:video_length]

        return query_top_k_indexes, results


def process_boxes(boxes, tgt_size, num_frames):
    boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    img_h, img_w = torch.tensor([tgt_size]).unbind(-1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes.cpu() * scale_fct[:, None, :]
    boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]], 0, img_w.item())
    boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]], 0, img_h.item())
    boxes = boxes.reshape(num_frames, -1, boxes.shape[-1]).detach()
    return boxes

@torch.no_grad()
def postprocess_softmax(num_frames, outputs, tgt_size, video_length):
    # end of model inference
    logits = outputs['pred_logits'].softmax(-1)[0, :, :-1]

    num_trajectories = outputs['pred_boxes'].shape[1] // num_frames

    pred_logits = logits.reshape(num_frames, num_trajectories, logits.shape[-1])


    pred_scores, pred_classes  = torch.max(pred_logits, dim=-1)
    pred_classes = pred_classes + 1

    results = {
        "scores": pred_scores[:video_length],
        "labels": pred_classes[:video_length],
    }

    boxes = outputs['pred_boxes'][0]
    pred_boxes = process_boxes(boxes, tgt_size, num_frames)
    results["boxes"] = pred_boxes[:video_length]

    return results


        # else:
        #     raise NotImplementedError
        #     # masks = outputs['pred_masks'][0]
        #     # pred_scores, pred_classes = pred_logits.max(-1)
        #     # if self.focal_loss:
        #     #     pred_classes = pred_classes + 1
        #     # masks = masks.reshape((self.num_frames, num_trajectories,) + masks.shape[-2:])
        #     #
        #     # results = {
        #     #     "scores": pred_scores[:video_length],
        #     #     "labels": pred_classes[:video_length],
        #     #     "boxes": pred_boxes[:video_length],
        #     #     "masks": masks[:video_length],
        #     # }
        #
        #     return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
