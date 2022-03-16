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
        constant_(self.sampling_offsets.weight.data, 0.) # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all
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

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
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
        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            #TODO: Not sure about this offset normalizer implementation
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # offset_normalizer =  torch.stack([input_spatial_shapes[..., 2], input_spatial_shapes[..., 0] * input_spatial_shapes[..., 1]], -1)

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

        return output

class TemporalDifferentModuleMSDeformAttn(nn.Module):

    def __init__(self, n_frames=36, d_model=256, n_levels=4, t_window=2, n_heads=8, n_curr_points=4, n_temporal_points=2, enc_use_new_sampling_init_default=False, for_decoder=False, embd_correlation=False, extra_linear_class=False, ):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """

        print(f"enc_use_new_sampling_init_default: {enc_use_new_sampling_init_default}")
        print(f"for_decoder: {for_decoder}")
        print(f"embd_correlation: {embd_correlation}")
        print(f"extra_linear_class: {extra_linear_class}")

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
        self.extra_linear_class = False
        assert not extra_linear_class
        self.embd_correlation = embd_correlation
        self.d_model = d_model
        self.n_frames = n_frames
        self.n_levels = n_levels
        self.t_window = t_window
        self.n_heads = n_heads
        self.n_curr_points = n_curr_points
        self.n_temporal_points = n_temporal_points
        self.enc_use_new_sampling_init_default = enc_use_new_sampling_init_default

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_curr_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_curr_points)

        self.temporal_sampling_offsets = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points * 2)
        self.temporal_attention_weights = nn.Linear(d_model, n_heads * n_levels * t_window * n_temporal_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.output_proj_class = None
        if self.extra_linear_class:
            assert self.for_decoder
            self.output_proj_class = nn.Linear(d_model, d_model)

        if self.embd_correlation:
            assert self.for_decoder

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.) # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all
        constant_(self.temporal_sampling_offsets.weight.data, 0.) # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all

        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])

        # curr_frame init
        curr_grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_curr_points, 1)
        for i in range(self.n_curr_points):
            curr_grid_init[:, :, i, :] *= i+1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(curr_grid_init.reshape(-1))

        # temporal init
        temporal_grid_init = grid_init.view(self.n_heads, 1, 1, 1, 2).repeat(1, self.n_levels, self.t_window, self.n_temporal_points, 1)
        if self.for_decoder and self.embd_correlation:
            for i in range(self.n_temporal_points):
                temporal_grid_init[:, :, :, i, :] *= i+1

        else:
            if self.enc_use_new_sampling_init_default:
                for i in range(self.n_temporal_points):
                    temporal_grid_init[:, :, :, i, :] *= i+1

            else:
                init_offset = 0
                for pos, t in enumerate(range(1, self.t_window + 1)):
                    if t % 2:
                        init_offset += 1
                    for i in range(self.n_temporal_points):
                        temporal_grid_init[:, :, pos, i, :] *= i + 1 + init_offset

                # for t in range(self.t_window // 2, self.t_window):
                #     for pos, i in enumerate(range(t + self.temporal_offset, t + self.temporal_offset + self.n_temporal_points)):
                #         temporal_grid_init[:, :, t, pos, :] *= i + 1
                #         temporal_grid_init[:, :, self.t_window-t - 1, pos, :] *= i + 1

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
        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.t_window, self.n_temporal_points, 2)
        temporal_attention_weights = self.temporal_attention_weights(query)
        temporal_attention_weights =  temporal_attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.t_window * self.n_temporal_points)

        curr_frame_attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_curr_points)

        attention_weights_curr_temporal = torch.cat([curr_frame_attention_weights, temporal_attention_weights], dim=3)
        attention_weights_curr_temporal =  F.softmax(attention_weights_curr_temporal, -1)

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
                # We need to normalize the reference point from the image [0,1] to the actual feature map that we have
                # We put all the reference points on the first and then we will apply the corresponding offset to each point in order to locate it on the corresponding frame on the global feature map
                curr_frame_offsets, other_frames_temporal_offsets = temporal_offsets
                reference_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)
                normalized_reference_points =  reference_points / reference_normalizer[None, None]
                projected_reference_points = normalized_reference_points + curr_frame_offsets[None]

                curr_frame_sampling_offsets = curr_frame_sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                temporal_sampling_offsets = other_frames_temporal_offsets[None, :, None, :, :, None, :] + temporal_sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]

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

            if self.embd_correlation:
                temporal_normalization = self.n_temporal_points
            else:
                temporal_normalization = self.t_window - 1 + self.temporal_offset + self.n_temporal_points

            if len(other_frames_temporal_offsets_xy) == 2:
                other_frames_temporal_offsets_xy_, other_frames_temporal_bbxes = other_frames_temporal_offsets_xy
                temporal_sampling_offsets = other_frames_temporal_offsets_xy_[None, :, None, :, :, None, :] +  \
                                            (temporal_sampling_offsets / temporal_normalization) * other_frames_temporal_bbxes[None, :, None, :, :, None, :] * 0.5

            else:
                temporal_sampling_offsets = other_frames_temporal_offsets_xy[None, :, None, :, :, None, :] +  \
                                            (temporal_sampling_offsets / temporal_normalization) * projected_reference_points[:, :, None, :, None, None, 2:] * 0.5

            sampling_temporal_offsets = temporal_sampling_offsets.flatten(4, 5)
            sampling_locations_offsets = torch.cat([curr_frame_sampling_offsets, sampling_temporal_offsets], dim=4)


            sampling_locations = projected_reference_points[:, :, None, :, None, :2] + sampling_locations_offsets

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        input_spatial_shapes_with_t = torch.stack((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1], input_spatial_shapes[:, 2]), dim=1)

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes_with_t, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        if self.output_proj_class is not None:
            output_for_class = output
            output = self.output_proj(output)
            output_for_class = self.output_proj_class(output_for_class)
            return output, output_for_class

        else:
            output = self.output_proj(output)
            return output

class TemporalDifferentModuleMSDeformAttnUnified(nn.Module):

    def __init__(self, n_frames=36, d_model=256, n_levels=4, n_heads=8, n_points=4, for_decoder=False, embd_correlation=False):
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
        self.embd_correlation = embd_correlation
        self.d_model = d_model
        self.n_frames = n_frames
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_frames * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_frames * n_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        if self.embd_correlation:
            assert self.for_decoder

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.) # sampling offset initialized weight to 0, so at initial iterations the bias is the only that matters at all

        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)

        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])

        # curr_frame init
        grid_init = grid_init.view(self.n_heads, 1, 1, 1,  2).repeat(1, self.n_levels, self.n_frames, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, :, i] *= i+1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1))

        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)


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

        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_frames * self.n_points)

        attention_weights =  F.softmax(attention_weights, -1)

        attention_weights = attention_weights.view(N, Len_q, self.n_heads, self.n_levels, self.n_frames, self.n_points)
        attention_weights = attention_weights.flatten(4, 5)

        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_frames, self.n_points, 2)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 2], input_spatial_shapes[..., 0] * input_spatial_shapes[..., 1]], -1)

            if self.for_decoder:
                curr_frame_offsets, temporal_offsets = temporal_offsets
                reference_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)
                normalized_reference_points =  reference_points / reference_normalizer[None, None]
                projected_reference_points = normalized_reference_points + curr_frame_offsets[None]

                sampling_offsets = temporal_offsets[None, :, None, :, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]
                sampling_locations_offsets = sampling_offsets.flatten(4, 5)

                sampling_locations = projected_reference_points[:, :, None, :, None, :] + sampling_locations_offsets


            else:
                sampling_locations = temporal_offsets[None, :, None, :, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]
                sampling_locations = sampling_locations.flatten(4, 5)
                sampling_locations = reference_points[:, :, None, :, None, :] + sampling_locations


        elif reference_points.shape[-1] == 4:
            assert self.for_decoder
            curr_frame_offsets_xy, other_frames_temporal_offsets_xy = temporal_offsets
            temporal_offsets_xy, temporal_modulation_xywh = other_frames_temporal_offsets_xy
            reference_normalizer_xy = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

            normalized_reference_points = reference_points[:, :, :, :2] / reference_normalizer_xy[None, None]
            projected_reference_points = normalized_reference_points + curr_frame_offsets_xy[None]

            sampling_offsets = temporal_offsets_xy[None, :, None, :, :, None, :] +  \
                                          (sampling_offsets / self.n_points) * temporal_modulation_xywh[None, :, None, :, :, None, :] * 0.5

            sampling_locations_offsets = sampling_offsets.flatten(4, 5)

            sampling_locations = projected_reference_points[:, :, None, :, None, :] + sampling_locations_offsets

        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        input_spatial_shapes_with_t = torch.stack((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1], input_spatial_shapes[:, 2]), dim=1)

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes_with_t, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        output = self.output_proj(output)
        return output


class MSDeformAttnPytorch(MSDeformAttn):

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
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
        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            #TODO: Not sure about this offset normalizer implementation
            offset_normalizer = torch.stack([input_spatial_shapes[..., 2], input_spatial_shapes[..., 0] * input_spatial_shapes[..., 1]], -1)

            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        #TODO: review this hacky implementation
        input_spatial_shapes_with_t = torch.stack((input_spatial_shapes[:,0] * input_spatial_shapes[:,1], input_spatial_shapes[:, 2]), dim=1)
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes_with_t, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return output

class TemporalDifferentModuleMSDeformAttnVIZ(TemporalDifferentModuleMSDeformAttn):



    def forward(self, query, reference_points, temporal_offsets, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, targets=None):
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

        layer = targets["layer"]
        value = self.value_proj(input_flatten)
        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        temporal_sampling_offsets = self.temporal_sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.t_window, self.n_temporal_points, 2)
        temporal_attention_weights = self.temporal_attention_weights(query)
        temporal_attention_weights =  temporal_attention_weights.view(N, Len_q, self.n_heads, self.n_levels * self.t_window * self.n_temporal_points)

        curr_frame_attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_curr_points)

        attention_weights_curr_temporal = torch.cat([curr_frame_attention_weights, temporal_attention_weights], dim=3)
        attention_weights_curr_temporal =  F.softmax(attention_weights_curr_temporal, -1)

        attention_weights_curr = attention_weights_curr_temporal[:, :, :, :self.n_levels * self.n_curr_points]
        attention_weights_temporal = attention_weights_curr_temporal[:, :, :, self.n_levels * self.n_curr_points:]

        attention_weights_curr = attention_weights_curr.view(N, Len_q, self.n_heads, self.n_levels, self.n_curr_points)

        targets[f"temporal_attention_weights_temporal_{layer}"] = torch.clone(attention_weights_temporal.view(N, Len_q, self.n_heads, self.n_levels, self.t_window, self.n_temporal_points))
        targets[f"attention_weights_curr_{layer}"] = torch.clone(attention_weights_curr.view(N, Len_q, self.n_heads, self.n_levels, self.n_curr_points))

        attention_weights_temporal = attention_weights_temporal.view(N, Len_q, self.n_heads, self.n_levels, self.t_window * self.n_temporal_points)


        attention_weights = torch.cat([attention_weights_curr, attention_weights_temporal], dim=4)


        curr_frame_sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_curr_points, 2)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 2], input_spatial_shapes[..., 0] * input_spatial_shapes[..., 1]], -1)

            if self.for_decoder:
                # We need to normalize the reference point from the image [0,1] to the actual feature map that we have
                # We put all the reference points on the first and then we will apply the corresponding offset to each point in order to locate it on the corresponding frame on the global feature map
                curr_frame_offsets, other_frames_temporal_offsets = temporal_offsets
                reference_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)
                normalized_reference_points =  reference_points / reference_normalizer[None, None]
                projected_reference_points = normalized_reference_points + curr_frame_offsets[None]

                curr_frame_sampling_offsets = curr_frame_sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                temporal_sampling_offsets = other_frames_temporal_offsets[None, :, None, :, :, None, :] + temporal_sampling_offsets / offset_normalizer[None, None, None, :, None, None, :]

                targets[f"sampling_temporal_offsets_{layer}"] = temporal_sampling_offsets + projected_reference_points[:, :, None, :, None, None,:]
                targets[f"current_sampling_offsets_{layer}"] = curr_frame_sampling_offsets + projected_reference_points[:, :, None, :, None, :]

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

            if self.embd_correlation:
                temporal_normalization = self.n_temporal_points
            else:
                temporal_normalization = self.t_window - 1 + self.temporal_offset + self.n_temporal_points

            if len(other_frames_temporal_offsets_xy) == 2:
                other_frames_temporal_offsets_xy_, other_frames_temporal_bbxes = other_frames_temporal_offsets_xy
                temporal_sampling_offsets = other_frames_temporal_offsets_xy_[None, :, None, :, :, None, :] +  \
                                            (temporal_sampling_offsets / temporal_normalization) * other_frames_temporal_bbxes[None, :, None, :, :, None, :] * 0.5

            else:
                temporal_sampling_offsets = other_frames_temporal_offsets_xy[None, :, None, :, :, None, :] +  \
                                            (temporal_sampling_offsets / temporal_normalization) * projected_reference_points[:, :, None, :, None, None, 2:] * 0.5


            targets[f"sampling_temporal_offsets_{layer}"] = temporal_sampling_offsets + projected_reference_points[:, :, None, :, None, None, :2]
            targets[f"current_sampling_offsets_{layer}"] = curr_frame_sampling_offsets + projected_reference_points[:, :, None, :, None, :2]

            sampling_temporal_offsets = temporal_sampling_offsets.flatten(4, 5)
            sampling_locations_offsets = torch.cat([curr_frame_sampling_offsets, sampling_temporal_offsets], dim=4)


            sampling_locations = projected_reference_points[:, :, None, :, None, :2] + sampling_locations_offsets



        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        input_spatial_shapes_with_t = torch.stack((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1], input_spatial_shapes[:, 2]), dim=1)

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes_with_t, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        if self.output_proj_class is not None:
            output_for_class = output
            output = self.output_proj(output)
            output_for_class = self.output_proj_class(output_for_class)
            return output, output_for_class

        else:
            output = self.output_proj(output)
            return output, targets