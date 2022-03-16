import random
import torch.nn as nn
import warnings
import torch
import torchvision
import torch.nn.functional as F
# from util.misc import match_name_keywords
from util.misc import NestedTensor, inverse_sigmoid
from .matcher import HungarianMatcher
from typing import List, Union
from torch import Tensor
from .deformable_vistr import AllPostProcessor, DeformableVisTR, MLP
from .deformable_transformer import DeformableTransformerDecoderLayer, DeformableTransformerDecoder
from .ops.modules import TemporalDifferentModuleMSDeformAttn
from collections import OrderedDict


ch_dict_en = {
    "/64": 256,
    "/32": 2048,
    "/16": 1024,
    "/8": 512,
    "/4": 256,
}

res_to_idx = {
    "/64": 3,
    "/32": 2,
    "/16": 1,
    "/8": 0,
}

backbone_res_to_idx = {
    "/32": 3,
    "/16": 2,
    "/8": 1,
    "/4": 0,
}

def compute_locations(h, w, device, stride=1):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device)

    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def compute_centroid_feature_map(higher_res, centroid_points):
    locations = compute_locations(higher_res[0], higher_res[1], device=centroid_points.device, stride=1)
    centroid_points = centroid_points * torch.tensor([higher_res[1],  higher_res[0]], device=centroid_points.device)
    locations = locations.reshape(1, 1, higher_res[0], higher_res[1], 2)
    centroids_reshape = centroid_points.reshape(centroid_points.shape[0], centroid_points.shape[1], 1, 1, 2)
    relative_coords = torch.abs(centroids_reshape - locations)

    x_norms = torch.max(centroid_points[:, :, 0], (higher_res[1]-1) - centroid_points[:, :, 0])
    y_norms = torch.max(centroid_points[:, :, 1], (higher_res[0]-1) - centroid_points[:, :, 1])
    norms = torch.stack([x_norms, y_norms], dim=-1)
    norms_relative_coords = 1 - relative_coords / norms[:,:, None, None]
    centroid_pos_feature = torch.pow((norms_relative_coords[..., 0] * norms_relative_coords[..., 1]), 4)
    return centroid_pos_feature


class DeformableVisTRsegmFinal(nn.Module):
    def __init__(self, vistr: DeformableVisTR, only_positive_matches: bool, matcher: HungarianMatcher,
                 mask_head_used_features: List[List[str]], att_maps_used_res: List[str], use_deformable_conv: bool,
                 post_processor: AllPostProcessor, top_k_inference: Union[int, None], mask_aux_loss: list, mask_head_out_mdcn:bool):

        super().__init__()
        self.vistr = vistr
        self.only_positive = only_positive_matches
        self.top_k_inference = top_k_inference
        self.matcher = matcher
        self.mask_aux_loss = mask_aux_loss
        self.mask_head_used_features = mask_head_used_features
        self.att_maps_used_res = att_maps_used_res
        self.postprocessor = post_processor
        self._sanity_check()
        feats_dims = self._get_mask_head_dims()
        hidden_dim, nheads = self.vistr.transformer.d_model, self.vistr.transformer.nhead_dec
        self.mask_head_hidden_size = hidden_dim // 16
        self.mask_embd_attention = DeformableAttSegmHead(hidden_dim, self.vistr.num_feature_levels, self.vistr.num_frames)
        self.bbox_attention = MultiScaleMHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0, num_levels=len(self.att_maps_used_res))
        self.mask_head = MaskHeadConv(hidden_dim, feats_dims, nheads, use_deformable_conv, self.att_maps_used_res, mask_head_out_mdcn)

        centroid_refine = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(centroid_refine.layers[-1].weight.data, 0)
        nn.init.constant_(centroid_refine.layers[-1].bias.data, 0)
        self.centroid_refine = centroid_refine

    def _sanity_check(self):
        init_mask_head_res, init_att_map_res = self.mask_head_used_features[0][0], self.att_maps_used_res[0]
        assert init_mask_head_res == init_att_map_res, f"Starting resolution for the mask_head_used features and att_maps_used_res has to be " \
                                                       f"the same. Got {init_mask_head_res} and {init_att_map_res} respectively"
        parent_class = [base.__name__ for base in self.__class__.__bases__]
        for cls in parent_class:
            if cls == "DETR":
                assert self.mask_head_used_features == [['/32', 'compressed_backbone'], ['/16', 'backbone'], ['/8', 'backbone'], ['/4', 'backbone']], \
                    "Only the following mask_head_used_features are available for DeTR: " \
                    "[['/32','compressed_backbone'], ['/16','backbone'], ['/8','backbone'], ['/4','backbone']]"
                assert self.att_maps_used_res == ['/32'], "Only the following mask head features are available for DeTR"

    def _get_mask_head_dims(self):
        feats_dims = []
        for res, name in self.mask_head_used_features[1:]:
            if name == "backbone":
                feats_dims.append(ch_dict_en[res])
            else:
                feats_dims.append(self.vistr.transformer.d_model)
        return feats_dims

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, src in enumerate(indices)])
        src_idx = torch.cat([src for src in indices])
        return batch_idx, src_idx


    def _get_features_for_mask_head(self, backbone_feats: List[Tensor], srcs: List[Tensor], memories: List[Tensor]):
        features_used = []
        for res, feature_type in self.mask_head_used_features:
            if feature_type == "backbone":
                if res == "/64":
                    warnings.warn("/64 feature map is only generated for encoded and compressed backbone feats. Using the compressed one")
                    features_used.append(srcs[res_to_idx[res]])
                else:
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors.unsqueeze(0).transpose(1,2))
            elif feature_type == "compressed_backbone":
                if res == "/4":
                    warnings.warn("/4 feature map is only generated for backbone. Using backbone")
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)
                else:
                    features_used.append(srcs[res_to_idx[res]])

            elif feature_type == "encoded":
                if res == "/4":
                    warnings.warn("/4 feature map is only generated for backbone. Using backbone")
                    features_used.append(backbone_feats[backbone_res_to_idx[res]].tensors)
                else:
                    features_used.append(memories[res_to_idx[res]])
            else:
                raise ValueError(
                    f"Selected feature type {feature_type} is not available. Available ones: [backbone, compressed_backbone, encoded]")
        return features_used

    def _get_training_embeddings(self, out, targets, hs, query_embed, reference_points, lvl):
        n_f = self.vistr.num_queries if  self.vistr.use_trajectory_queries else self.vistr.num_queries // self.vistr.num_frames
        if not self.only_positive:
            raise NotImplementedError
            # num_trajectories = n_f
            # hs_f = hs[lvl][0].view(self.vistr.num_frames, num_trajectories, hs.shape[-1])
            # query_embed_f = query_embed[0].view(self.vistr.num_frames, num_trajectories, hs.shape[-1])
            # reference_points = reference_points[0].view(self.vistr.num_frames, num_trajectories, hs.shape[-1])

        else:
            outputs_without_aux = {k: v for k, v in out.items() if k != 'aux_outputs' and k != 'enc_outputs' and k != 'hs_embed'}
            indices, embd_matched_idx, valid_ind = self.matcher(outputs_without_aux, targets, return_embd_idx=True)
            if embd_matched_idx is None:
                # Generate temporal mask for 1 instance, which will be thrown away later on
                embd_matched_idx = [torch.arange(self.vistr.num_frames, device=hs.device) * n_f]

            num_trajectories = len(embd_matched_idx)
            matched_hs = []
            matched_query_embed = []
            matched_ref_points = []
            for idx in embd_matched_idx:
                matched_hs.append(hs[lvl][0, idx])
                matched_query_embed.append(query_embed[0, idx])
                matched_ref_points.append(reference_points[lvl][0, idx])
            hs_f = torch.stack(matched_hs, dim=1)
            query_embed_f = torch.stack(matched_query_embed, dim=1)
            reference_points_f = torch.stack(matched_ref_points, dim=1)

            out["indices"] = indices
            out["mask_valid_ind"] = valid_ind

        return hs_f, query_embed_f, reference_points_f, num_trajectories

    def _get_eval_top_k_embeddings(self, out, targets, hs, query_embed, reference_points):
        process_boxes = targets["process_boxes"] if "process_boxes" in targets else True
        top_k_idxs, results = self.postprocessor(out, targets["tgt_size"], targets["clip_length"], process_boxes)
        top_k_embeddings, inverse_idxs = torch.unique(top_k_idxs, return_inverse=True)
        num_trajectories = top_k_embeddings.shape[0]
        hs_f = hs[:, top_k_embeddings]
        query_embed_f = query_embed[:, top_k_embeddings]
        reference_points_f = reference_points[:, top_k_embeddings]

        return  hs_f, query_embed_f, reference_points_f, num_trajectories, results, inverse_idxs

    def _centroid_refine(self, hs_f, reference_points):
        matched_reference_points = inverse_sigmoid(reference_points)
        centroid_offset = self.centroid_refine(hs_f)
        centroid_offset = matched_reference_points + centroid_offset
        centroid_points = centroid_offset.sigmoid()
        return centroid_points

    def _module_inference(self, hs_f, query_embed, memory_flatten, memories_att_map_f, masks_att_map, mask_head_feats_f, num_trajectories,
                          spatial_shapes, reference_points, valid_ratios, level_start_index):

        centroid_points = self._centroid_refine(hs_f, reference_points)
        hs_f = self.mask_embd_attention(memory_flatten, hs_f, query_embed, spatial_shapes, centroid_points, valid_ratios, level_start_index)
        bbox_mask_f = self.bbox_attention(hs_f, memories_att_map_f, mask=masks_att_map)
        bbox_mask_flattened = [bbox_mask.transpose(1, 0).flatten(0, 1) for bbox_mask in bbox_mask_f]
        higher_res = mask_head_feats_f[-1].shape[-2:]
        centroid_pos_feature = compute_centroid_feature_map(higher_res, centroid_points)
        centroid_pos_feature = centroid_pos_feature.transpose(1, 0).flatten(0, 1)[:, None]
        seg_masks_f = self.mask_head(mask_head_feats_f, centroid_pos_feature, bbox_mask_flattened, instances_per_batch=num_trajectories)
        outputs_seg_masks = seg_masks_f.view((num_trajectories, self.vistr.num_frames) + seg_masks_f.shape[2:])
        centroid_points = centroid_points.transpose(1, 0)
        return outputs_seg_masks, centroid_points

    def forward(self, samples: NestedTensor, targets: dict):
        out, backbone_feats, memories_flatten, memories, hs, query_pos, srcs, masks, init_reference, inter_references, \
        level_start_index, valid_ratios, spatial_shapes = self.vistr(samples, targets)

        if not isinstance(memories, list):
            memories_att_map, masks_att_map = [memories], [masks]
        else:
            memories_att_map = [memories[res_to_idx[res]] for res in self.att_maps_used_res]
            masks_att_map = [masks[res_to_idx[res]] for res in self.att_maps_used_res]

        # Take
        if not isinstance(srcs, list):
            mask_head_feats = [srcs, backbone_feats[2].tensors, backbone_feats[1].tensors, backbone_feats[0].tensors]
        else:
            mask_head_feats = self._get_features_for_mask_head(backbone_feats, srcs, memories)

        # image level processing using box attention
        memories_att_map_f = [memory[0].transpose(0, 1) for memory in memories_att_map]
        mask_head_feats_f = [feat[0].transpose(0, 1) for feat in mask_head_feats]

        if self.training:
            loss_levels = [-1] + self.mask_aux_loss
            for loss_lvl in loss_levels:
                out_lvl = out if loss_lvl == -1 else out["aux_outputs"][loss_lvl]
                hs_f, query_embed_f, reference_points_f, num_trajectories = self._get_training_embeddings(out_lvl, targets, hs, query_pos, inter_references, loss_lvl)
                output_seg_masks, outputs_centroids = self._module_inference(hs_f, query_embed_f, memories_flatten, memories_att_map_f, masks_att_map, mask_head_feats_f,
                                                           num_trajectories, spatial_shapes, reference_points_f, valid_ratios, level_start_index)

                out_lvl["pred_masks"] = output_seg_masks.flatten(0, 1)
                out_lvl["centroid_points"] = outputs_centroids.flatten(0, 1)

            return out

        else:
            reference_points = inter_references[-1]
            out["reference_points"] = reference_points
            num_trajectories = self.vistr.num_queries if self.vistr.use_trajectory_queries else self.vistr.num_queries // self.vistr.num_frames
            hs_f = hs[-1][0].view(self.vistr.num_frames, num_trajectories, hs.shape[-1])
            query_pos_f = query_pos[0].view(self.vistr.num_frames, num_trajectories, query_pos.shape[-1])
            reference_points_f = reference_points[0].view(self.vistr.num_frames, num_trajectories, reference_points.shape[-1])

            if self.top_k_inference is not None:
                hs_f, query_embed_f, reference_points_f, num_trajectories, results, inverse_idxs = self._get_eval_top_k_embeddings(out, targets, hs_f, query_pos_f, reference_points_f)
                outputs_seg_masks, centroid_points = self._module_inference(hs_f, query_embed_f, memories_flatten, memories_att_map_f, masks_att_map, mask_head_feats_f,
                                                           num_trajectories, spatial_shapes, reference_points_f, valid_ratios, level_start_index)
                out_masks = outputs_seg_masks.transpose(0, 1)
                results["masks"] = out_masks[:targets["clip_length"]]
                results["centroid_points"] = centroid_points.transpose(0, 1)
                results["inverse_idxs"] = inverse_idxs
                results["centroid_prediction"] = True

            else:
                raise NotImplementedError
                # outputs_seg_masks, centroid_points = self._module_inference(hs_f, query_pos, memories_flatten, memories_att_map_f, masks_att_map, mask_head_feats_f,
                #                                            num_trajectories, spatial_shapes, reference_points, valid_ratios, level_start_index)
                #
                # out["pred_masks"] = outputs_seg_masks.transpose(0, 1).flatten(0, 1).unsqueeze(0)
                # process_boxes = targets["process_boxes"] if "process_boxes" in targets else True
                # results = self.postprocessor(out, targets["tgt_size"], targets["clip_length"], process_boxes)
                # results["centroid_points"] = centroid_points

            return results

class ModulatedDeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ModulatedDeformableConv2d, self).__init__()

        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride,
                                        padding=self.padding, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=self.padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias,
                                          padding=self.padding, mask=modulator)
        return x


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)


def expand_multi_length(tensor, lengths):
    if isinstance(lengths, list):
        return tensor.unsqueeze(1).repeat(1, int(lengths[0]), 1, 1, 1).flatten(0, 1)
    else:
        return tensor.repeat(lengths, 1, 1, 1)


class MultiScaleMHAttentionMap(nn.Module):

    def __init__(self, query_dim, hidden_dim, num_heads, num_levels, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        for i in range(num_levels):
            layer_name = "" if i == 0 else f"_{i}"
            setattr(self, f"q_linear{layer_name}", nn.Linear(query_dim, hidden_dim, bias=bias))
            setattr(self, f"k_linear{layer_name}", nn.Linear(query_dim, hidden_dim, bias=bias))
            nn.init.zeros_(getattr(self, f"k_linear{layer_name}").bias)
            nn.init.zeros_(getattr(self, f"q_linear{layer_name}").bias)
            nn.init.xavier_uniform_(getattr(self, f"k_linear{layer_name}").weight)
            nn.init.xavier_uniform_(getattr(self, f"q_linear{layer_name}").weight)

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def _check_input(self, k, mask):
        assert len(k) == self.num_levels
        if mask is not None:
            assert len(mask) == self.num_levels

    def forward(self, q, k, mask=None):
        self._check_input(k, mask)
        out_multi_scale_maps = []

        for i, k_lvl in enumerate(k):
            layer_name = "" if i == 0 else f"_{i}"
            q_lvl = q
            q_lvl = getattr(self, f"q_linear{layer_name}")(q_lvl)
            k_lvl = F.conv2d(k_lvl, getattr(self, f"k_linear{layer_name}").weight.unsqueeze(-1).unsqueeze(-1),
                             getattr(self, f"k_linear{layer_name}").bias)
            qh_lvl = q_lvl.view(q_lvl.shape[0], q_lvl.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
            kh_lvl = k_lvl.view(k_lvl.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k_lvl.shape[-2], k_lvl.shape[-1])
            weights = torch.einsum("bqnc,bnchw->bqnhw", qh_lvl * self.normalize_fact, kh_lvl)
            if mask is not None:
                weights.masked_fill_(mask[i].unsqueeze(1).unsqueeze(1), float("-inf"))
            weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
            # weights = self.dropout(weights)
            out_multi_scale_maps.append(weights)

        return out_multi_scale_maps


class MaskHeadConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, nheads, use_deformable_conv, multi_scale_att_maps, use_out_mdcn=False, extra_channels=1):
        super().__init__()

        num_levels = len(fpn_dims) + 1
        out_dims = [dim // (2 ** exp) for exp in range(num_levels + 2)]
        in_dims = [dim // (2 ** exp) for exp in range(num_levels + 2)]
        for i in range(len(multi_scale_att_maps)):
            in_dims[i] += nheads

        self.multi_scale_att_maps = len(multi_scale_att_maps) > 1
        conv_layer = ModulatedDeformableConv2d if use_deformable_conv else Conv2d

        self.lay1 = conv_layer(in_dims[0], in_dims[0], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, in_dims[0])

        self.lay2 = conv_layer(in_dims[0], out_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, out_dims[1])

        for i in range(1, num_levels):
            if i == num_levels -1:
                setattr(self, f"lay{i + 2}", conv_layer(in_dims[i] + extra_channels, out_dims[i + 1], 3, padding=1))
                setattr(self, f"gn{i + 2}", torch.nn.GroupNorm(8, out_dims[i + 1]))
            else:
                setattr(self, f"lay{i + 2}", conv_layer(in_dims[i], out_dims[i + 1], 3, padding=1))
                setattr(self, f"gn{i + 2}", torch.nn.GroupNorm(8, out_dims[i + 1]))
            setattr(self, f"adapter{i}", Conv2d(fpn_dims[i - 1], out_dims[i], 1, padding=0))

        if use_out_mdcn:
            self.out_lay = nn.Sequential(OrderedDict([
                ("out_lay1", conv_layer(out_dims[i + 1], out_dims[i + 2], 3, padding=1)),
                ("out_gn1", torch.nn.GroupNorm(8, out_dims[i + 2])),
                ("relu", nn.ReLU()),
                ("out_lay2", conv_layer(out_dims[i + 2], 1, 3, padding=1)),
            ]))

        else:
            self.out_lay = Conv2d(out_dims[i + 1], 1, 3, padding=1)

    def forward(self, features, ct_heatmap, bbox_mask, instances_per_batch):

        expanded_feats = expand_multi_length(features[0], instances_per_batch)
        x = torch.cat([expanded_feats, bbox_mask[0]], 1)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        for lvl, feature in enumerate(features[1:]):
            cur_fpn = getattr(self, f"adapter{lvl + 1}")(feature)
            cur_fpn = expand_multi_length(cur_fpn, instances_per_batch)
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
            if self.multi_scale_att_maps and lvl + 1 < len(bbox_mask):
                x = torch.cat([x, bbox_mask[lvl + 1]], 1)
            else:
                if  len(x.shape)== len(ct_heatmap.shape):
                    x = torch.cat([x, ct_heatmap], 1)
                else:
                    x = torch.cat([x, ct_heatmap[:, None]], 1)
            x = getattr(self, f"lay{lvl + 3}")(x)
            x = getattr(self, f"gn{lvl + 3}")(x)
            x = F.relu(x)
        x = self.out_lay(x)

        return x


class DeformableDecoderCustomLayer(DeformableTransformerDecoderLayer):

    def forward(self, tgt, query_pos, reference_points, curr_frame_offsets, other_frames_temporal_offsets, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0].transpose(0, 1)
        tgt = tgt.transpose(0, 1) + self.dropout2(tgt2)
        tgt = self.norm2(tgt)


        # cross attention
        tgt = tgt.transpose(0, 1).flatten(0, 1)[None]
        query_pos = query_pos.flatten(0, 1)[None]
        curr_temporal_offset = (curr_frame_offsets, other_frames_temporal_offsets)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, curr_temporal_offset,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableAttSegmHead(nn.Module):
    def __init__(self, d_model, num_feature_levels, num_frames):
        super().__init__()
        self.dec_connect_all_embeddings = True
        self.dec_embedding_correlation = True
        self.num_frames = num_frames
        self.temporal_window = self.num_frames - 1
        dim_feedforward = 1024
        dropout = 0.1
        activation= "relu"
        nhead_dec = 8
        dec_n_curr_points = 2
        dec_n_temporal_points = 1

        self.module = DeformableDecoderCustomLayer(d_model, dim_feedforward,
                                                          dropout, activation, num_frames, self.temporal_window,
                                                          num_feature_levels, nhead_dec, dec_n_curr_points, dec_n_temporal_points, self.dec_embedding_correlation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, TemporalDifferentModuleMSDeformAttn):
                m._reset_parameters()

    def forward(self, src, tgt_masks, query_pos, src_spatial_shapes, reference_points, src_valid_ratios, src_level_start_index):
        reference_points = reference_points.flatten(0, 1)[None]

        num_frames, embds_per_frame, _ = tgt_masks.shape
        curr_temporal_offset, other_temporal_offsets = DeformableTransformerDecoder.get_curr_frame_and_temporal_offsets_decoder_all_connect(src_spatial_shapes,
                                                                                                                        embds_per_frame,
                                                                                                                        src_valid_ratios, tgt_masks.device)

        other_frames_temporal_offsets = DeformableTransformerDecoder.get_temporal_offset_from_same_instance_all_connect(reference_points, other_temporal_offsets,
                                                                                                src_spatial_shapes, src_valid_ratios, False)

        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] \
                                     * torch.cat([src_valid_ratios[0, None], src_valid_ratios[0, None]], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * src_valid_ratios[0, None]

        hs_mask = self.module(tgt_masks, query_pos, reference_points_input, curr_temporal_offset, other_frames_temporal_offsets, src, src_spatial_shapes,
                       src_level_start_index, None)
        hs_mask = hs_mask[0].reshape([num_frames, embds_per_frame, hs_mask.shape[-1]])

        return hs_mask
