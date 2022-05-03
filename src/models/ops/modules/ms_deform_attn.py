# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction, ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)

        return output, None


# TODO: Docstring for this class
class TemporalMSDeformAttnBase(nn.Module):

    def __init__(self, n_frames=36, d_model=256, n_levels=4, t_window=2, n_heads=8, n_curr_points=4, n_temporal_points=2, dec_instance_aware_att=True, for_decoder=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level

        """

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.temporal_offset = 0
        self.for_decoder = for_decoder
        self.dec_instance_aware_att = dec_instance_aware_att
        self.d_model = d_model
        self.n_frames = n_frames
        self.n_levels = n_levels
        self.t_window = t_window
        self.n_heads = n_heads
        self.n_curr_points = n_curr_points
        self.n_temporal_points = n_temporal_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_curr_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_curr_points)

        self.temporal_sampling_offsets = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points * 2)
        self.temporal_attention_weights = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)  # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all
        constant_(self.temporal_sampling_offsets.weight.data, 0.)  # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all

        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])

        # curr_frame init
        curr_grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_curr_points, 1)
        for i in range(self.n_curr_points):
            curr_grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(curr_grid_init.reshape(-1))

        # temporal init
        temporal_grid_init = grid_init.view(self.n_heads, 1, 1, 1, 2).repeat(1, self.n_levels, self.t_window, self.n_temporal_points, 1)

        for i in range(self.n_temporal_points):
            temporal_grid_init[:, :, :, i, :] *= i + 1

        with torch.no_grad():
            self.temporal_sampling_offsets.bias = nn.Parameter(temporal_grid_init.reshape(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        constant_(self.temporal_attention_weights.weight.data, 0.)
        constant_(self.temporal_attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def _compute_deformable_attention(self, query, input_flatten):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        T_, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(T_, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(T_, Len_q, self.n_heads, self.t_window, self.n_levels, self.n_temporal_points, 2)
        temporal_sampling_offsets = temporal_sampling_offsets.flatten(3, 4)

        temporal_attention_weights = self.temporal_attention_weights(query)
        temporal_attention_weights = temporal_attention_weights.view(T_, Len_q, self.n_heads, self.t_window * self.n_levels * self.n_temporal_points)

        curr_frame_attention_weights = self.attention_weights(query).view(T_, Len_q, self.n_heads, self.n_levels * self.n_curr_points)

        attention_weights_curr_temporal = torch.cat([curr_frame_attention_weights, temporal_attention_weights], dim=3)
        attention_weights_curr_temporal = F.softmax(attention_weights_curr_temporal, -1)

        attention_weights_curr = attention_weights_curr_temporal[:, :, :, :self.n_levels * self.n_curr_points]
        attention_weights_temporal = attention_weights_curr_temporal[:, :, :, self.n_levels * self.n_curr_points:]

        attention_weights_curr = attention_weights_curr.view(T_, Len_q, self.n_heads, self.n_levels, self.n_curr_points).contiguous()
        attention_weights_temporal = attention_weights_temporal.view(T_, Len_q, self.n_heads, self.t_window * self.n_levels, self.n_temporal_points).contiguous()

        curr_frame_sampling_offsets = self.sampling_offsets(query).view(T_, Len_q, self.n_heads, self.n_levels, self.n_curr_points, 2)

        return value, curr_frame_sampling_offsets, temporal_sampling_offsets, attention_weights_curr, attention_weights_temporal


class TemporalMSDeformAttnDecoder(TemporalMSDeformAttnBase):

    def __init__(self, n_frames=36, d_model=256, n_levels=4, t_window=2, n_heads=8, n_curr_points=4, n_temporal_points=2, dec_instance_aware_att=True):
        super().__init__(n_frames=n_frames, d_model=d_model, n_levels=n_levels, t_window=t_window, n_heads=n_heads, n_curr_points=n_curr_points,
                         n_temporal_points=n_temporal_points)
        self.dec_instance_aware_att = dec_instance_aware_att

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, temporal_offsets):

        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index

        T_ = input_flatten.shape[0]
        embd_per_frame = query.shape[1] // T_
        query = query.reshape([T_, embd_per_frame, query.shape[-1]])

        if reference_points.shape[0] != T_:
            reference_points = reference_points.reshape((T_, embd_per_frame) + reference_points.shape[-2:])

        T_, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape

        value, curr_frame_sampling_offsets, temporal_sampling_offsets, attention_weights_curr, \
            attention_weights_temporal = super()._compute_deformable_attention(query, input_flatten)

        # To add hook for att maps visualization
        current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps = [], []
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
            temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

            for t in range(T_):
                current_frame_values = value[t][None]
                sampling_locations = reference_points[t][None, :, None, :, None] \
                                     + curr_frame_sampling_offsets[t][None] / offset_normalizer[None, None, None, :, None, :]

                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes, input_current_level_start_index,
                    sampling_locations, attention_weights_curr[t][None], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]

                if self.dec_instance_aware_att:
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(1, 2)[None, :, None, :, None]
                else:
                    temporal_ref_points = reference_points[t].repeat(1, self.t_window, 1)[None, :, None, :, None]

                temporal_sampling_locations = temporal_ref_points \
                                              + temporal_sampling_offsets[t][None] / temporal_offsets_normalizer[None, None, None, :, None, :]

                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes, input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t][None], self.im2col_step)

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        elif reference_points.shape[-1] == 4:
            for t in range(T_):
                current_frame_values = value[t][None]
                sampling_locations = reference_points[t][None, :, None, :, None, :2] \
                                     + (curr_frame_sampling_offsets[t][None] / self.n_curr_points) * reference_points[t][None, :, None, :, None, 2:] * 0.5

                current_sampling_locations_for_att_maps.append(sampling_locations)
                output_curr = MSDeformAttnFunction.apply(
                    current_frame_values, input_current_spatial_shapes,
                    input_current_level_start_index, sampling_locations,
                    attention_weights_curr[t][None], self.im2col_step
                )

                temporal_frames = temporal_offsets[t] + t
                temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]

                if self.dec_instance_aware_att:
                    temporal_ref_points = reference_points[temporal_frames].transpose(0, 1).flatten(1, 2)[None, :, None, :, None]
                else:
                    temporal_ref_points = reference_points[t].repeat(1, self.t_window, 1)[None, :, None, :, None]

                temporal_sampling_locations = temporal_ref_points[:, :, :, :, :, :2] \
                                              + (temporal_sampling_offsets[t][None] / self.n_temporal_points) * temporal_ref_points[:, :, :, :, :, 2:] * 0.5

                temporal_sampling_locations_for_att_maps.append(temporal_sampling_locations)
                output_temporal = MSDeformAttnFunction.apply(
                    temporal_frames_values, input_temporal_spatial_shapes,
                    input_temporal_level_start_index, temporal_sampling_locations,
                    attention_weights_temporal[t][None], self.im2col_step
                )

                frame_output = output_curr + output_temporal
                output.append(frame_output)

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        output = torch.cat(output, dim=0).flatten(0, 1)[None]
        output = self.output_proj(output)

        return output, current_sampling_locations_for_att_maps, temporal_sampling_locations_for_att_maps, attention_weights_curr, attention_weights_temporal


class TemporalMSDeformAttnEncoder(TemporalMSDeformAttnBase):

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, temporal_offsets):
        output = []
        input_current_spatial_shapes, input_temporal_spatial_shapes = input_spatial_shapes
        input_current_level_start_index, input_temporal_level_start_index = input_level_start_index
        T_, Len_q, _ = query.shape
        T_, Len_in, _ = input_flatten.shape
        assert reference_points.shape[-1] == 2

        value, curr_frame_sampling_offsets, temporal_sampling_offsets, attention_weights_curr, attention_weights_temporal = super()._compute_deformable_attention(query,
                                                                                                                                                                  input_flatten)

        offset_normalizer = torch.stack([input_current_spatial_shapes[..., 1], input_current_spatial_shapes[..., 0]], -1)
        temporal_offsets_normalizer = offset_normalizer.repeat(self.t_window, 1)

        for t in range(T_):
            current_frame_values = value[t][None]
            sampling_locations = reference_points[t][None, :, None, :, None] \
                                 + curr_frame_sampling_offsets[t][None] / offset_normalizer[None, None, None, :, None, :]

            output_curr = MSDeformAttnFunction.apply(
                current_frame_values, input_current_spatial_shapes, input_current_level_start_index, sampling_locations, attention_weights_curr[t][None], self.im2col_step)

            temporal_frames = temporal_offsets[t] + t
            temporal_frames_values = value[temporal_frames].flatten(0, 1)[None]
            temporal_ref_points = reference_points[t, :, 0][None, :, None, None, None]

            temporal_sampling_locations = temporal_ref_points \
                                          + temporal_sampling_offsets[t][None] / temporal_offsets_normalizer[None, None, None, :, None, :]

            output_temporal = MSDeformAttnFunction.apply(
                temporal_frames_values, input_temporal_spatial_shapes, input_temporal_level_start_index, temporal_sampling_locations,
                attention_weights_temporal[t][None], self.im2col_step)

            frame_output = output_curr + output_temporal
            output.append(frame_output)

        output = torch.cat(output, dim=0)
        output = self.output_proj(output)
        return output, None


class TemporalFlattenMSDeformAttn(TemporalMSDeformAttnBase):

    def forward(self, query, reference_points, temporal_offsets, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1] * input_spatial_shapes[:, 2]).sum() == Len_in

        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.t_window, self.n_temporal_points, 2)
        temporal_attention_weights = self.temporal_attention_weights(query)
        temporal_attention_weights = temporal_attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.t_window * self.n_temporal_points)

        curr_frame_attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_curr_points)

        attention_weights_curr_temporal = torch.cat([curr_frame_attention_weights, temporal_attention_weights], dim=3)
        attention_weights_curr_temporal = F.softmax(attention_weights_curr_temporal, -1)

        attention_weights_curr = attention_weights_curr_temporal[:, :, :, :self.n_levels * self.n_curr_points]
        attention_weights_temporal = attention_weights_curr_temporal[:, :, :, self.n_levels * self.n_curr_points:]

        attention_weights_curr = attention_weights_curr.view(N, Len_q, self.n_heads, self.n_levels, self.n_curr_points)
        attention_weights_temporal = attention_weights_temporal.view(N, Len_q, self.n_heads, self.n_levels, self.t_window * self.n_temporal_points)
        attention_weights = torch.cat([attention_weights_curr, attention_weights_temporal], dim=4)

        curr_frame_sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_curr_points, 2)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 2], input_spatial_shapes[..., 0] * input_spatial_shapes[..., 1]], -1)

            if self.for_decoder:
                # We need to normalize the reference point from the image [0,1] to the actual feature map that we have We put all the reference points on the first and then we
                # will apply the corresponding offset to each point in order to locate it on the corresponding frame on the global feature map
                curr_frame_offsets, other_frames_temporal_offsets = temporal_offsets
                reference_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)
                normalized_reference_points = reference_points / reference_normalizer[None, None]
                projected_reference_points = normalized_reference_points + curr_frame_offsets[None]

                curr_frame_sampling_offsets = curr_frame_sampling_offsets / offset_normalizer[None, None, None, :, None, :]

                temporal_sampling_offsets = other_frames_temporal_offsets[None, :, None, :, :, None, :] + \
                                            temporal_sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]

                sampling_temporal_offsets = temporal_sampling_offsets.flatten(4, 5)
                sampling_locations_offsets = torch.cat([curr_frame_sampling_offsets, sampling_temporal_offsets], dim=4)

                sampling_locations = projected_reference_points[:, :, None, :, None, :] + sampling_locations_offsets

            else:
                sampling_temporal_offsets = temporal_offsets[None, :, None, :, :, None, :] + temporal_sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]
                sampling_temporal_offsets = sampling_temporal_offsets.flatten(4, 5)

                curr_frame_sampling_offsets = curr_frame_sampling_offsets / offset_normalizer[None, None, None, :, None, :]

                sampling_locations = torch.cat([curr_frame_sampling_offsets, sampling_temporal_offsets], dim=4)
                sampling_locations = reference_points[:, :, None, :, None, :] + sampling_locations

        elif reference_points.shape[-1] == 4:
            assert self.for_decoder
            curr_frame_offsets_xy, other_frames_temporal_offsets_xy = temporal_offsets
            reference_normalizer_xy = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)
            reference_normalizer_xywh = torch.cat([reference_normalizer_xy[0, None], reference_normalizer_xy[0, None]], -1)

            curr_frame_offsets_wh = torch.zeros_like(curr_frame_offsets_xy)
            curr_frame_offsets_xywh = torch.cat([curr_frame_offsets_xy, curr_frame_offsets_wh], dim=-1)

            normalized_reference_points = reference_points / reference_normalizer_xywh[None, None]
            projected_reference_points = normalized_reference_points + curr_frame_offsets_xywh[None]

            curr_frame_sampling_offsets = (curr_frame_sampling_offsets / self.n_curr_points) * projected_reference_points[:, :, None, :, None, 2:] * 0.5
            # TODO: rename variables
            if len(other_frames_temporal_offsets_xy) == 2:
                other_frames_temporal_offsets_xy_, other_frames_temporal_bbxes = other_frames_temporal_offsets_xy
                temporal_sampling_offsets = other_frames_temporal_offsets_xy_[None, :, None, :, :, None, :] + \
                                            (temporal_sampling_offsets / self.n_temporal_points) * other_frames_temporal_bbxes[None, :, None, :, :, None, :] * 0.5

            else:
                temporal_sampling_offsets = other_frames_temporal_offsets_xy[None, :, None, :, :, None, :] + \
                                            (temporal_sampling_offsets / self.n_temporal_points) * projected_reference_points[:, :, None, :, None, None, 2:] * 0.5

            sampling_temporal_offsets = temporal_sampling_offsets.flatten(4, 5)
            sampling_locations_offsets = torch.cat([curr_frame_sampling_offsets, sampling_temporal_offsets], dim=4)

            sampling_locations = projected_reference_points[:, :, None, :, None, :2] + sampling_locations_offsets

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        input_spatial_shapes_with_t = torch.stack((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1], input_spatial_shapes[:, 2]), dim=1)

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes_with_t, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        output = self.output_proj(output)
        return output
