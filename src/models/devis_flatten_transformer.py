# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from src.util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn, TemporalMSDeformAttn
from .deformable_transformer import _get_clones, _get_activation_fn


class DeVISTransformer(nn.Module):
    def __init__(self, d_model=256, num_frames=6, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, activation="relu",
                 num_feature_levels=4, enc_connect_all_embeddings=True, enc_temporal_window=2, enc_n_curr_points=4, enc_n_temporal_points=2,
                 dec_n_curr_points=4, dec_n_temporal_points=2, dec_instance_aware_att=True, instance_level_queries=False, with_gradient=False):

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_lvls = num_feature_levels
        self.instance_level_queries = instance_level_queries
        self.with_gradient = with_gradient
        self.num_frames = num_frames

        if enc_connect_all_embeddings:
            enc_temporal_window = num_frames - 1

        encoder_layer = DeVISTransformerEncoderLayer(d_model, dim_feedforward,
                                                     dropout, activation, num_frames, enc_temporal_window,
                                                     num_feature_levels, nhead, enc_n_curr_points, enc_n_temporal_points)

        self.encoder = DeVISTransformerEncoder(encoder_layer, num_encoder_layers, enc_temporal_window, enc_connect_all_embeddings)

        dec_temporal_window = num_frames - 1
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, num_frames, dec_temporal_window,
                                                          num_feature_levels, nhead, dec_n_curr_points, dec_n_temporal_points)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, dec_temporal_window, with_gradient, dec_instance_aware_att)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, (MSDeformAttn, TemporalMSDeformAttn)):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, targets=None):
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            n, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.reshape(n // self.num_frames, self.num_frames, c, h, w).permute(0, 2, 1, 3, 4).flatten(-3)
            src_flatten.append(src)
            mask = mask.reshape(n // self.num_frames, self.num_frames, h, w).flatten(-3)
            mask_flatten.append(mask)
            pos_embed = pos_embed.permute(0, 2, 1, 3, 4).flatten(-3)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, -1, 1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        src_flatten = torch.cat(src_flatten, 2).permute(0, 2, 1)
        mask_flatten = torch.cat(mask_flatten, 1).unsqueeze(-1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2).permute(0, 2, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        spatial_shapes = torch.cat((torch.tensor([[self.num_frames]], device=spatial_shapes.device).repeat(self.num_lvls, 1), spatial_shapes), dim=1)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.instance_level_queries:
            query_embed = query_embed.repeat(self.num_frames, 1)
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        # decoder
        hs, inter_references, _ = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index,
            valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references

        offset = 0
        memories = []
        for src in srcs:
            num_frames, c, h, w = src.shape
            memory_slice = memory[:, offset:offset + h * w * num_frames].permute(0, 2, 1).reshape(1, c, num_frames, h, w)
            memories.append(memory_slice)
            offset += h * w * num_frames

        return hs, query_embed, memories, init_reference_out, inter_references_out, level_start_index, valid_ratios, spatial_shapes


class DeVISTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_frames=6, t_window=2, n_levels=4, n_heads=8, n_curr_points=4, n_temporal_points=2):
        super().__init__()
        self.self_attn = TemporalMSDeformAttn(n_frames, d_model, n_levels, t_window, n_heads, n_curr_points, n_temporal_points, False)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, temporal_offsets, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, temporal_offsets, src, spatial_shapes, level_start_index, padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeVISTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, t_window, enc_connect_all_embeddings):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.t_window = t_window
        self.enc_connect_all_embeddings = enc_connect_all_embeddings

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (T_, H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ * T_ - 0.5, H_ * T_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[0, None, lvl, 1] * H_ * T_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[0, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[0, None, None]
        return reference_points

    @staticmethod
    def generate_temporal_offsets_all_connect(input_spatial_shapes, valid_ratios, device):
        temporal_offsets = []

        for lvl in range(input_spatial_shapes.shape[0]):
            T_, H_, W_ = input_spatial_shapes[lvl]
            temporal_per_spatial_offset = []

            for curr_frame in range(0, T_):
                temporal_frames = torch.tensor([[0, t / T_] for t in range(-curr_frame, T_ - curr_frame) if t != 0], device=device)
                temporal_frames = temporal_frames[None].repeat((W_ * H_, 1, 1))
                temporal_per_spatial_offset.append(temporal_frames)

            temporal_offsets.append(torch.cat(temporal_per_spatial_offset, dim=0))

        temporal_offsets = torch.cat(temporal_offsets, dim=0)
        temporal_offsets = temporal_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        return temporal_offsets

    # TODO: Refactor this ugly function
    @staticmethod
    def generate_temporal_offsets_all_connect_higher_clip_size(input_spatial_shapes, temporal_window, valid_ratios, device):
        temporal_offsets = []
        # We will have 3 different scenarios
        # 1) First frame of the clip: We will sample past frame as future frames reversed
        temporal_frames_first = [t for t in range((temporal_window // 2) + 1) if t != 0]
        temporal_frames_first = temporal_frames_first[::-1] + temporal_frames_first

        # 2) Last frame of the clip: We will sample future frames as past frames reversed
        temporal_frames_last = [-t for t in range((temporal_window // 2) + 1) if t != 0]
        temporal_frames_last = temporal_frames_last[::-1] + temporal_frames_last

        # 3) In-between frames: Follow the sliding window approach across each one
        temporal_frames = [t for t in range(-temporal_window // 2, (temporal_window // 2) + 1) if t != 0]

        for lvl in range(input_spatial_shapes.shape[0]):
            T_, H_, W_ = input_spatial_shapes[lvl]
            temporal_per_spatial_offset = []

            # Special case first frame
            frame_offsets = [torch.tensor([0, t_2 / T_], device=device) for t_2 in temporal_frames_first]
            temporal_per_spatial_offset.append(torch.stack(frame_offsets, dim=0).repeat((W_ * H_, 1, 1)))

            for t_1 in range(1, T_ - 1):
                frame_offsets = []
                for t_2 in temporal_frames:
                    # Padding scenario 1) Pad with the offset referring to first frame of sequence
                    if t_1 + t_2 < 0:
                        frame_offsets.append(torch.tensor([0, -t_2 / T_], device=device))

                    # Padding scenario 1) Pad with the offset referring to last frame of sequence
                    elif t_1 + t_2 > T_ - 1:
                        frame_offsets.append(torch.tensor([0, -t_2 / T_], device=device))

                    else:
                        frame_offsets.append(torch.tensor([0, t_2 / T_], device=device))

                temporal_per_spatial_offset.append(torch.stack(frame_offsets, dim=0).repeat((W_ * H_, 1, 1)))

            # Special case last frame
            frame_offsets = [torch.tensor([0, t_2 / T_], device=device) for t_2 in temporal_frames_last]
            temporal_per_spatial_offset.append(torch.stack(frame_offsets, dim=0).repeat((W_ * H_, 1, 1)))

            temporal_offsets.append(torch.cat(temporal_per_spatial_offset, dim=0))

        temporal_offsets = torch.cat(temporal_offsets, dim=0)
        temporal_offsets = temporal_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        return temporal_offsets

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        if self.enc_connect_all_embeddings:
            temporal_offsets = self.generate_temporal_offsets_all_connect(spatial_shapes, valid_ratios, device=src.device)
        else:
            temporal_offsets = self.generate_temporal_offsets_all_connect_higher_clip_size(spatial_shapes, self.t_window, valid_ratios, device=src.device)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, temporal_offsets, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_frames=36, t_window=2, n_levels=4, n_heads=8, n_curr_points=4, n_temporal_points=2):
        super().__init__()
        self.cross_attn = TemporalMSDeformAttn(n_frames, d_model, n_levels, t_window, n_heads, n_curr_points, n_temporal_points, True)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, curr_frame_offsets, other_frames_temporal_offsets, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        curr_temporal_offset = (curr_frame_offsets, other_frames_temporal_offsets)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, curr_temporal_offset,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, temporal_window, instance_aware_att=True, with_gradient=False):

        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.instance_aware_att = instance_aware_att
        self.temporal_window = temporal_window
        self.with_gradient = with_gradient
        self.return_intermediate = True

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.ref_point_embed = None

    @staticmethod
    def get_curr_frame_and_temporal_offsets_decoder(num_frames, embds_per_frame, valid_ratios, device):
        curr_frame_offsets, temporal_offsets = [], []
        for curr_frame in range(0, num_frames):
            curr_frame_offsets.append(torch.tensor([0, curr_frame / num_frames], device=device).repeat(embds_per_frame, 1))
            temporal_frames = torch.tensor([[0, t / num_frames] for t in range(-curr_frame, num_frames - curr_frame) if t != 0], device=device)
            temporal_frames = temporal_frames[None].repeat(embds_per_frame, 1, 1)
            temporal_offsets.append(temporal_frames)

        curr_frame_offsets = torch.cat(curr_frame_offsets, dim=0)
        curr_frame_offsets = curr_frame_offsets[:, None, :] * valid_ratios[0, None]

        temporal_offsets = torch.cat(temporal_offsets, dim=0)
        temporal_offsets = temporal_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        return curr_frame_offsets, temporal_offsets

    @staticmethod
    def get_instance_aware_temporal_offsets(reference_points, temporal_offsets, num_frames, valid_ratios):
        T_ = num_frames
        embds_per_frame = reference_points.shape[1] // T_.item()

        temporal_offsets_per_frame = temporal_offsets[:, 0].reshape(T_, embds_per_frame, temporal_offsets.shape[-2], temporal_offsets.shape[-1])
        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        total_bbx_modulation = []

        distance_normalizer = torch.tensor([[1, T_]], device=reference_points.device)

        for curr_frame in range(0, T_):
            temporal_frames = curr_frame + torch.tensor([t for t in range(-curr_frame, T_ - curr_frame) if t != 0], device=reference_points.device)
            if reference_points_per_frame.shape[-1] == 4:
                reference_points_distance = reference_points_per_frame[temporal_frames, :, :2] - reference_points_per_frame[curr_frame, :, :2][None]
            else:
                reference_points_distance = reference_points_per_frame[temporal_frames] - reference_points_per_frame[curr_frame][None]

            reference_points_distance = reference_points_distance.transpose(0, 1)
            total_offset_distance = temporal_offsets_per_frame[curr_frame] + reference_points_distance / distance_normalizer[None]
            total_offsets.append(total_offset_distance)

            if reference_points_per_frame.shape[-1] == 4:
                temporal_alignments = reference_points_per_frame[temporal_frames, :, 2:] / distance_normalizer
                total_bbx_modulation.append(temporal_alignments.transpose(0, 1))

        total_offsets = torch.cat(total_offsets, dim=0)
        total_offsets = total_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        if total_bbx_modulation:
            total_bbx_modulation = torch.cat(total_bbx_modulation, dim=0)
            total_bbx_modulation = total_bbx_modulation[:, None, :, :] * valid_ratios[0, None, :, None]
            return total_offsets, total_bbx_modulation

        return total_offsets

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):

        output = tgt
        intermediate = []
        intermediate_reference_points = []
        num_frames = src_spatial_shapes[0][0]

        embds_per_frame = torch.div(reference_points.shape[1], num_frames, rounding_mode='trunc')
        curr_frame_offset, temporal_frames_offsets_base = self.get_curr_frame_and_temporal_offsets_decoder(num_frames,
                                                                                                           embds_per_frame,
                                                                                                           src_valid_ratios, tgt.device)

        temporal_frames_offsets = torch.clone(temporal_frames_offsets_base)

        for lid, layer in enumerate(self.layers):
            if self.instance_aware_att:
                temporal_frames_offsets = self.get_instance_aware_temporal_offsets(reference_points, temporal_frames_offsets_base, num_frames, src_valid_ratios)

            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios[0, None], src_valid_ratios[0, None]], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[0, None]

            output = layer(output, query_pos, reference_points_input, curr_frame_offset, temporal_frames_offsets, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                if self.with_gradient:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()

            if self.ref_point_embed is not None:
                tmp = self.ref_point_embed[lid](output)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                reference_points = new_reference_points.sigmoid()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), None

        return output, reference_points, None


def build_devis_transformer(cfg):
    return DeVISTransformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD,
        dropout=cfg.MODEL.DROPOUT,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        with_gradient=cfg.MODEL.BBX_GRADIENT_PROP,

        num_encoder_layers=cfg.MODEL.TRANSFORMER.ENCODER_LAYERS,
        num_decoder_layers=cfg.MODEL.TRANSFORMER.DECODER_LAYERS,
        nhead=cfg.MODEL.TRANSFORMER.N_HEADS,
        enc_n_curr_points=cfg.MODEL.TRANSFORMER.ENC_N_POINTS,
        dec_n_curr_points=cfg.MODEL.TRANSFORMER.DEC_N_POINTS,

        num_frames=cfg.MODEL.DEVIS.NUM_FRAMES,
        enc_connect_all_embeddings=cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_CONNECT_ALL_FRAMES,
        enc_temporal_window=cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_TEMPORAL_WINDOW,
        enc_n_temporal_points=cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.ENC_N_POINTS_TEMPORAL_FRAME,
        dec_instance_aware_att=cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.INSTANCE_AWARE_ATTENTION,
        dec_n_temporal_points=cfg.MODEL.DEVIS.DEFORMABLE_ATTENTION.DEC_N_POINTS_TEMPORAL_FRAME,
        instance_level_queries=cfg.MODEL.DEVIS.INSTANCE_LEVEL_QUERIES,

        activation="relu",

    )
