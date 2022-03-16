# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import math

import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from util.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn, TemporalDifferentModuleMSDeformAttn, TemporalDifferentModuleMSDeformAttnVIZ
from .transformer import _get_clones, _get_activation_fn

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, num_frames=6, nhead_enc=8, nhead_dec=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, enc_temporal_window=2, dec_temporal_window=2, dec_connect_all_embeddings=False, dec_embedding_correlation=False, dec_embedding_correlation_alignment=False,
                 dec_n_curr_points=4,  enc_n_curr_points=4, dec_n_temporal_points=2, enc_n_temporal_points=2,
                 two_stage=False, two_stage_num_proposals=300, use_trajectory_queries=False, with_gradient=False, with_decoder_instance_self_attn=False,
                 with_decoder_frame_self_attn=False, enc_connect_all_embeddings=False, dec_sort_temporal_offsets=False, enc_use_new_sampling_init_default=False):

        super().__init__()

        self.d_model = d_model
        self.num_lvls = num_feature_levels
        self.use_trajectory_queries = use_trajectory_queries
        self.with_gradient = with_gradient
        self.num_frames = num_frames
        self.nhead_enc = nhead_enc
        self.nhead_dec = nhead_dec

        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        new_higher_clip_size_offsets = False
        # if enc_connect_all_embeddings and self.num_frames == 36:
        #     print(f"new_higher_clip_size_offsets {new_higher_clip_size_offsets}")
        #     enc_temporal_window = enc_temporal_window
        #     new_higher_clip_size_offsets = True

        if enc_connect_all_embeddings:
            enc_temporal_window = self.num_frames - 1

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, num_frames, enc_temporal_window,
                                                          num_feature_levels, nhead_enc, enc_n_curr_points, enc_n_temporal_points, enc_use_new_sampling_init_default)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, enc_temporal_window, enc_connect_all_embeddings, new_higher_clip_size_offsets)

        if dec_connect_all_embeddings:
            # assert dec_embedding_correlation
            dec_temporal_window = self.num_frames - 1



        if with_decoder_instance_self_attn or with_decoder_frame_self_attn:
            if with_decoder_frame_self_attn:
                assert not with_decoder_instance_self_attn

            decoder_layer = DeformableTransformerDecoderLayerExtraSelfAttn(d_model, dim_feedforward,
                                                              dropout, activation, num_frames, dec_temporal_window,
                                                              num_feature_levels, nhead_dec, dec_n_curr_points, dec_n_temporal_points,
                                                              dec_embedding_correlation, with_decoder_instance_self_attn, with_decoder_frame_self_attn)
        else:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                              dropout, activation, num_frames, dec_temporal_window,
                                                              num_feature_levels, nhead_dec, dec_n_curr_points, dec_n_temporal_points, dec_embedding_correlation)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, dec_embedding_correlation, dec_embedding_correlation_alignment, dec_connect_all_embeddings, num_frames,  dec_temporal_window, return_intermediate_dec, with_gradient, dec_sort_temporal_offsets)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, (MSDeformAttn, TemporalDifferentModuleMSDeformAttn, TemporalDifferentModuleMSDeformAttnVIZ)):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio


    def forward(self, srcs, masks, pos_embeds, query_embed=None, targets=None):
        assert self.two_stage or query_embed is not None

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
        spatial_shapes = torch.cat((torch.tensor([[self.num_frames]], device=spatial_shapes.device).repeat(self.num_lvls, 1), spatial_shapes),
                                   dim=1)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            if self.use_trajectory_queries:
                query_embed = query_embed.repeat(self.num_frames, 1)
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references, targets = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index,
            valid_ratios, query_embed, mask_flatten, targets)

        inter_references_out = inter_references
        return hs, query_embed, memory, init_reference_out, inter_references_out, level_start_index, valid_ratios, spatial_shapes, targets


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_frames=6, t_window=2,  n_levels=4, n_heads=8, n_curr_points=4, n_temporal_points=2, enc_use_new_sampling_init_default=False):

        super().__init__()
        self.self_attn = TemporalDifferentModuleMSDeformAttn(n_frames, d_model, n_levels, t_window, n_heads, n_curr_points, n_temporal_points, enc_use_new_sampling_init_default, False, False)

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


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, t_window, enc_connect_all_embeddings, new_higher_clip_size_offsets):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.t_window = t_window
        self.enc_connect_all_embeddings = enc_connect_all_embeddings
        self.new_higher_clip_size_offsets = new_higher_clip_size_offsets

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
       reference_points_list = []
       for lvl, (T_, H_, W_) in enumerate(spatial_shapes):
           ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_*T_ - 0.5, H_ * T_, dtype=torch.float32, device=device),
                                         torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
           ref_y = ref_y.reshape(-1)[None] / (valid_ratios[0, None, lvl, 1] * H_ * T_)
           ref_x = ref_x.reshape(-1)[None] / (valid_ratios[0, None, lvl, 0] * W_)
           ref = torch.stack((ref_x, ref_y), -1)
           reference_points_list.append(ref)
       reference_points = torch.cat(reference_points_list, 1)
       reference_points = reference_points[:, :, None] * valid_ratios[0, None, None]
       return reference_points




    @staticmethod
    def generate_temporal_offsets(input_spatial_shapes, temporal_window, valid_ratios, device):
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

            for t_1 in range(1, T_-1):
                frame_offsets = []
                for t_2 in temporal_frames:
                    # Padding scenario 1) Pad with the offset referring to first frame of sequence
                    if t_1 + t_2 < 0:
                        frame_offsets.append(torch.tensor([0, -t_1 / T_], device=device))

                    # Padding scenario 1) Pad with the offset referring to last frame of sequence
                    elif t_1 + t_2 > T_ - 1:
                        frame_offsets.append(torch.tensor([0, (T_-1-t_1) / T_], device=device))

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

            for t_1 in range(1, T_-1):
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
            if self.new_higher_clip_size_offsets:
                temporal_offsets = self.generate_temporal_offsets_all_connect_higher_clip_size(spatial_shapes, self.t_window, valid_ratios, device=src.device)
            else:
                temporal_offsets = self.generate_temporal_offsets_all_connect(spatial_shapes, valid_ratios, device=src.device)

        else:
            temporal_offsets = self.generate_temporal_offsets(spatial_shapes, self.t_window, valid_ratios, device=src.device)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, temporal_offsets, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_frames=36, t_window=2, n_levels=4, n_heads=8, n_curr_points=4, n_temporal_points=2, dec_embedding_correlation=False):

        super().__init__()
        self.cross_attn = TemporalDifferentModuleMSDeformAttnVIZ(n_frames, d_model, n_levels, t_window, n_heads, n_curr_points, n_temporal_points, False, True, dec_embedding_correlation)

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

    def forward(self, tgt, query_pos, reference_points, curr_frame_offsets, other_frames_temporal_offsets, src, src_spatial_shapes, level_start_index, src_padding_mask=None, targets=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        curr_temporal_offset = (curr_frame_offsets, other_frames_temporal_offsets)
        tgt2, targets = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, curr_temporal_offset,
                               src, src_spatial_shapes, level_start_index, src_padding_mask, targets)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, targets


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, dec_embedding_correlation, dec_embedding_correlation_alignment, dec_connect_all_embeddings,
                 num_frames, temporal_window, return_intermediate=False, with_gradient=False, sort_temporal_offsets=False):
        super().__init__()
        if sort_temporal_offsets:
            assert dec_connect_all_embeddings
        self.dec_connect_all_embeddings = dec_connect_all_embeddings
        self.dec_embedding_correlation_alignment = dec_embedding_correlation_alignment
        self.dec_embedding_correlation = dec_embedding_correlation
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.with_gradient = with_gradient
        self.sort_temporal_offsets = sort_temporal_offsets

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.ref_point_embed = None
        self.num_frames = num_frames
        self.temporal_window = temporal_window

    @staticmethod
    def get_curr_frame_and_temporal_offsets_decoder(input_spatial_shapes, num_embds, temporal_window, valid_ratios, device):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = num_embds // T_
        curr_frame_offsets = []
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


        for t_1 in range(0, T_):
            curr_frame_offsets.append(torch.tensor([0, t_1 / T_], device=device).repeat(embds_per_frame, 1))
            if t_1 == 0:
                # Special case first frame
                frame_offsets = [torch.tensor([0, t_2 / T_], device=device) for t_2 in temporal_frames_first]

            elif t_1 == T_ - 1:
                frame_offsets = [torch.tensor([0, t_2 / T_], device=device) for t_2 in temporal_frames_last]

            else:
                frame_offsets = []
                for t_2 in temporal_frames:
                    # Padding scenario 1) Pad with the offset referring to first frame of sequence
                    if t_1 + t_2 < 0:
                        frame_offsets.append(torch.tensor([0, -t_2 / T_], device=device))
                        # frame_offsets.append(torch.tensor([0, -t_1 / T_], device=device))

                    # Padding scenario 1) Pad with the offset referring to last frame of sequence
                    elif t_1 + t_2 > T_ - 1:
                        frame_offsets.append(torch.tensor([0, -t_2 / T_], device=device))
                        # frame_offsets.append(torch.tensor([0, (T_ - 1 - t_1) / T_], device=device))

                    else:
                        frame_offsets.append(torch.tensor([0, t_2 / T_], device=device))

            temporal_offsets.append(torch.stack(frame_offsets, dim=0).repeat((embds_per_frame, 1, 1)))

        curr_frame_offsets = torch.cat(curr_frame_offsets, dim=0)
        curr_frame_offsets = curr_frame_offsets[:, None, :]  * valid_ratios[0, None]

        temporal_offsets = torch.cat(temporal_offsets, dim=0)
        temporal_offsets = temporal_offsets[:, None, :, :] * valid_ratios[0, None, :, None]


        return curr_frame_offsets, temporal_offsets


    @staticmethod
    def get_curr_frame_and_temporal_offsets_decoder_all_connect(input_spatial_shapes, embds_per_frame, valid_ratios, device):
        T_ = input_spatial_shapes[0][0]
        curr_frame_offsets = []
        temporal_offsets = []

        # TODO: Explore performance putting offset from closer to further frame
        for t_1 in range(0, T_):
            frame_offsets = []
            for t_2 in range(-t_1, T_ - t_1):
                if t_2 != 0:
                    frame_offsets.append(torch.tensor([0, t_2 / T_], device=device))
                else:
                    curr_frame_offsets.append(torch.tensor([0, t_1 / T_], device=device).repeat(embds_per_frame, 1))

            temporal_offsets.append(torch.stack(frame_offsets, dim=0).repeat((embds_per_frame, 1, 1)))

        curr_frame_offsets = torch.cat(curr_frame_offsets, dim=0)
        curr_frame_offsets = curr_frame_offsets[:, None, :]  * valid_ratios[0, None]

        temporal_offsets = torch.cat(temporal_offsets, dim=0)
        temporal_offsets = temporal_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        return curr_frame_offsets, temporal_offsets

    @staticmethod
    def get_curr_frame_and_temporal_offsets_decoder_all_connect_parallel(input_spatial_shapes, embds_per_frame, valid_ratios, device):
        T_ = input_spatial_shapes[0][0]
        curr_frame_offsets = []
        temporal_offsets = []

        # TODO: Explore performance putting offset from closer to further frame
        for curr_frame in range(0, T_):
            curr_frame_offsets.append(torch.tensor([0, curr_frame / T_], device=device).repeat(embds_per_frame, 1))
            temporal_frames = torch.tensor([[0, t / T_] for t in range(-curr_frame, T_ - curr_frame) if t != 0], device=device)
            temporal_frames = temporal_frames[None].repeat(embds_per_frame, 1, 1)
            temporal_offsets.append(temporal_frames)

        curr_frame_offsets = torch.cat(curr_frame_offsets, dim=0)
        curr_frame_offsets = curr_frame_offsets[:, None, :]  * valid_ratios[0, None]

        temporal_offsets = torch.cat(temporal_offsets, dim=0)
        temporal_offsets = temporal_offsets[:, None, :, :] * valid_ratios[0, None, :, None]


        return curr_frame_offsets, temporal_offsets



    @staticmethod
    def get_curr_frame_and_temporal_offsets_decoder_all_connect_sorted(input_spatial_shapes, embds_per_frame, valid_ratios, device):
        T_ = input_spatial_shapes[0][0]
        curr_frame_offsets = []
        temporal_offsets_per_embedding, temporal_offsets = [], []
        temporal_positions_per_frame = []

        for t_1 in range(0, T_):
            curr_frame_offsets.append(torch.tensor([0, t_1 / T_], device=device).repeat(embds_per_frame, 1))
            frame_offsets = []
            temporal_positions = []
            t_2 = 1
            while len(frame_offsets) < T_ - 1:
                if t_1 + t_2 < T_:
                    frame_offsets.append(torch.tensor([0, t_2 / T_], device=device))
                    temporal_positions.append(t_2)
                if t_1 - t_2 >= 0:
                    frame_offsets.append(torch.tensor([0, -t_2 / T_], device=device))
                    temporal_positions.append(-t_2)
                t_2 += 1

            temporal_offsets_per_embedding.append(torch.stack(frame_offsets, dim=0).repeat((embds_per_frame, 1, 1)))
            # temporal_offsets.append(torch.stack(frame_offsets, dim=0))
            temporal_positions_per_frame.append(temporal_positions)

        curr_frame_offsets = torch.cat(curr_frame_offsets, dim=0)
        curr_frame_offsets = curr_frame_offsets[:, None, :]  * valid_ratios[0, None]

        # temporal_offsets = torch.stack(temporal_offsets, dim=0)

        temporal_offsets_per_embedding = torch.cat(temporal_offsets_per_embedding, dim=0)
        temporal_offsets_per_embedding = temporal_offsets_per_embedding[:, None, :, :] * valid_ratios[0, None, :, None]

        return curr_frame_offsets, temporal_offsets_per_embedding, temporal_positions_per_frame


    @staticmethod
    def get_temporal_offset_from_same_instance(reference_points, temporal_offsets, input_spatial_shapes, temporal_window, valid_ratios, device):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = reference_points.shape[1] // T_

        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        temporal_offsets = temporal_offsets[:, 0]
        distance_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

        # temporal_offsets = []
        temporal_frames = [t for t in range(-temporal_window // 2, (temporal_window // 2) + 1) if t != 0]
        for t_1 in range(0, T_):
            frame_offsets, frame_valid = [], []

            for pos, t_2 in enumerate(temporal_frames):

                if t_1 + t_2 < 0 or t_1 + t_2 > T_ - 1:
                    frame_offsets.append(torch.tensor([0, 0], device=device).repeat(embds_per_frame, 1))
                else:
                    if reference_points_per_frame.shape[-1] == 4:
                        reference_points_distance = reference_points_per_frame[t_1+t_2, :, :2] - reference_points_per_frame[t_1, :, :2]
                    else:
                        assert reference_points.shape[-1] == 2
                        reference_points_distance = reference_points_per_frame[t_1+t_2]  - reference_points_per_frame[t_1]

                    total_offset_distance = temporal_offsets[t_1*embds_per_frame:(t_1+1)*embds_per_frame, pos] + reference_points_distance / distance_normalizer
                    frame_offsets.append(total_offset_distance)

            total_offsets.append(torch.stack(frame_offsets, dim=1))

        total_offsets = torch.cat(total_offsets, dim=0)
        total_offsets = total_offsets[:, None, :, :] * valid_ratios[0, None, :, None]
        return total_offsets







    @staticmethod
    def get_temporal_offset_from_same_instance_all_connect(reference_points, temporal_offsets, input_spatial_shapes, valid_ratios, dec_embedding_correlation_alignment):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = reference_points.shape[1] // T_

        temporal_offsets_per_frame = temporal_offsets[:, 0].reshape(T_, embds_per_frame, temporal_offsets.shape[-2], temporal_offsets.shape[-1])
        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        total_alignment = []

        distance_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

        for t_1 in range(0, T_):
            frame_offsets = []
            frame_alignments = []
            temporal_frames = [t for t in range(-t_1, T_ - t_1) if t != 0]
            for pos, t_2 in enumerate(temporal_frames):
                if reference_points_per_frame.shape[-1] == 4:
                    reference_points_distance = reference_points_per_frame[t_1 + t_2, :, :2] - reference_points_per_frame[t_1, :, :2]
                    if dec_embedding_correlation_alignment:
                        frame_alignments.append(reference_points_per_frame[t_1 + t_2, :, 2:] / distance_normalizer)
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_distance = reference_points_per_frame[t_1 + t_2] - reference_points_per_frame[t_1]

                total_offset_distance = temporal_offsets_per_frame[t_1, :, pos] + reference_points_distance / distance_normalizer
                frame_offsets.append(total_offset_distance)

            total_offsets.append(torch.stack(frame_offsets, dim=1))
            if frame_alignments and dec_embedding_correlation_alignment:
                total_alignment.append(torch.stack(frame_alignments, dim=1))


        total_offsets = torch.cat(total_offsets, dim=0)
        total_offsets = total_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        if total_alignment and dec_embedding_correlation_alignment:
            total_alignment = torch.cat(total_alignment, dim=0)
            total_alignment = total_alignment[:, None, :, :] * valid_ratios[0, None, :, None]
            return total_offsets, total_alignment

        return total_offsets


    @staticmethod
    def get_temporal_offset_from_same_instance_all_connect_parallel(reference_points, temporal_offsets, input_spatial_shapes, valid_ratios, dec_embedding_correlation_alignment):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = reference_points.shape[1] // T_

        temporal_offsets_per_frame = temporal_offsets[:, 0].reshape(T_, embds_per_frame, temporal_offsets.shape[-2], temporal_offsets.shape[-1])
        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        total_alignment = []

        distance_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

        for curr_frame in range(0, T_):
            temporal_frames = curr_frame + torch.tensor([t for t in range(-curr_frame, T_ - curr_frame) if t != 0], device=reference_points.device)

            if reference_points_per_frame.shape[-1] == 4:
                reference_points_distance =  reference_points_per_frame[temporal_frames, :, :2] - reference_points_per_frame[curr_frame, :, :2][None]
            else:
                reference_points_distance =  reference_points_per_frame[temporal_frames] - reference_points_per_frame[curr_frame][None]

            reference_points_distance = reference_points_distance.transpose(0, 1)
            total_offset_distance = temporal_offsets_per_frame[curr_frame] + reference_points_distance / distance_normalizer[None]
            total_offsets.append(total_offset_distance)

            if dec_embedding_correlation_alignment and reference_points_per_frame.shape[-1] == 4:
                temporal_alignments = reference_points_per_frame[temporal_frames, :, 2:] / distance_normalizer
                total_alignment.append(temporal_alignments.transpose(0, 1))
                # total_alignment.append(torch.stack(frame_alignments, dim=1))

        total_offsets = torch.cat(total_offsets, dim=0)
        total_offsets = total_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        if total_alignment and dec_embedding_correlation_alignment:
            total_alignment = torch.cat(total_alignment, dim=0)
            total_alignment = total_alignment[:, None, :, :] * valid_ratios[0, None, :, None]
            return total_offsets, total_alignment

        return total_offsets

    @staticmethod
    def get_temporal_offset_from_same_instance_all_connect_sorted_offsets(reference_points, temporal_offsets, temporal_positions_per_frame, input_spatial_shapes, valid_ratios, dec_embedding_correlation_alignment):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = reference_points.shape[1] // T_

        temporal_offsets_per_frame = temporal_offsets[:, 0].reshape(T_, embds_per_frame, temporal_offsets.shape[-2], temporal_offsets.shape[-1])
        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        total_alignment = []

        distance_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

        for t_1 in range(0, T_):
            frame_offsets = []
            frame_alignments = []
            anchor_frame = t_1
            for pos, t_2 in enumerate(temporal_positions_per_frame[t_1]):
                temporal_frame = anchor_frame + t_2
                temporal_offset_wrt_anchor = temporal_offsets_per_frame[t_1, :, pos]
                if reference_points_per_frame.shape[-1] == 4:
                    reference_points_distance = reference_points_per_frame[temporal_frame, :, :2] - reference_points_per_frame[anchor_frame, :, :2]
                    if dec_embedding_correlation_alignment:
                        frame_alignments.append(reference_points_per_frame[temporal_frame, :, 2:] / distance_normalizer)
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_distance = reference_points_per_frame[temporal_frame] - reference_points_per_frame[anchor_frame]

                total_offset_distance = temporal_offset_wrt_anchor + reference_points_distance / distance_normalizer
                frame_offsets.append(total_offset_distance)

            total_offsets.append(torch.stack(frame_offsets, dim=1))
            if frame_alignments and dec_embedding_correlation_alignment:
                total_alignment.append(torch.stack(frame_alignments, dim=1))


        total_offsets = torch.cat(total_offsets, dim=0)
        total_offsets = total_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        if total_alignment and dec_embedding_correlation_alignment:
            total_alignment = torch.cat(total_alignment, dim=0)
            total_alignment = total_alignment[:, None, :, :] * valid_ratios[0, None, :, None]
            return total_offsets, total_alignment

        return total_offsets


    @staticmethod
    def get_temporal_offset_from_same_instance_all_connect_sorted_offsets_parallel(reference_points, temporal_offsets, temporal_positions_per_frame, input_spatial_shapes, valid_ratios, dec_embedding_correlation_alignment):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = reference_points.shape[1] // T_

        temporal_offsets_per_frame = temporal_offsets[:, 0].reshape(T_, embds_per_frame, temporal_offsets.shape[-2], temporal_offsets.shape[-1])
        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        total_alignment = []

        distance_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

        for t_1 in range(0, T_):
            curr_frame = t_1
            temporal_frames = curr_frame + torch.tensor(temporal_positions_per_frame[t_1], device=reference_points.device)

            if reference_points_per_frame.shape[-1] == 4:
                reference_points_distance = reference_points_per_frame[temporal_frames, :, :2] - reference_points_per_frame[curr_frame, :, :2][None]
            else:
                reference_points_distance = reference_points_per_frame[temporal_frames] - reference_points_per_frame[curr_frame][None]

            reference_points_distance = reference_points_distance.transpose(0, 1)
            total_offset_distance = temporal_offsets_per_frame[curr_frame] + reference_points_distance / distance_normalizer[None]
            total_offsets.append(total_offset_distance)

            if dec_embedding_correlation_alignment and reference_points_per_frame.shape[-1] == 4:
                temporal_alignments = reference_points_per_frame[temporal_frames, :, 2:] / distance_normalizer
                total_alignment.append(temporal_alignments.transpose(0, 1))

        total_offsets = torch.cat(total_offsets, dim=0)
        total_offsets = total_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        if total_alignment and dec_embedding_correlation_alignment:
            total_alignment = torch.cat(total_alignment, dim=0)
            total_alignment = total_alignment[:, None, :, :] * valid_ratios[0, None, :, None]
            return total_offsets, total_alignment

        return total_offsets


    @staticmethod
    def get_temporal_offset_from_same_instance_all_connect_parallel(reference_points, temporal_offsets, input_spatial_shapes, valid_ratios, dec_embedding_correlation_alignment):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = reference_points.shape[1] // T_

        temporal_offsets_per_frame = temporal_offsets[:, 0].reshape(T_, embds_per_frame, temporal_offsets.shape[-2], temporal_offsets.shape[-1])
        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        total_alignment = []

        distance_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

        for curr_frame in range(0, T_):
            temporal_frames = curr_frame + torch.tensor([t for t in range(-curr_frame, T_ - curr_frame) if t != 0], device=reference_points.device)

            if reference_points_per_frame.shape[-1] == 4:
                reference_points_distance =  reference_points_per_frame[temporal_frames, :, :2] - reference_points_per_frame[curr_frame, :, :2][None]
            else:
                reference_points_distance =  reference_points_per_frame[temporal_frames] - reference_points_per_frame[curr_frame][None]

            reference_points_distance = reference_points_distance.transpose(0, 1)
            total_offset_distance = temporal_offsets_per_frame[curr_frame] + reference_points_distance / distance_normalizer[None]
            total_offsets.append(total_offset_distance)

            if dec_embedding_correlation_alignment and reference_points_per_frame.shape[-1] == 4:
                temporal_alignments = reference_points_per_frame[temporal_frames, :, 2:] / distance_normalizer
                total_alignment.append(temporal_alignments.transpose(0, 1))
                # total_alignment.append(torch.stack(frame_alignments, dim=1))

        total_offsets = torch.cat(total_offsets, dim=0)
        total_offsets = total_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        if total_alignment and dec_embedding_correlation_alignment:
            total_alignment = torch.cat(total_alignment, dim=0)
            total_alignment = total_alignment[:, None, :, :] * valid_ratios[0, None, :, None]
            return total_offsets, total_alignment

        return total_offsets

    @staticmethod
    def get_temporal_offset_from_same_instance_parallel(reference_points, temporal_window, temporal_offsets, input_spatial_shapes, valid_ratios, dec_embedding_correlation_alignment):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = reference_points.shape[1] // T_
        num_frames = T_.item()
        temporal_offsets_per_frame = temporal_offsets[:, 0].reshape(T_, embds_per_frame, temporal_offsets.shape[-2], temporal_offsets.shape[-1])
        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        total_alignment = []

        distance_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

        general_temporal_frames = [t for t in range(-temporal_window // 2, (temporal_window // 2) + 1) if t != 0]

        for curr_frame in range(0, T_):
            temporal_offsets_curr_frames = []
            for t_2 in general_temporal_frames:
                if t_2 + curr_frame < 0 or t_2 + curr_frame > num_frames - 1:
                    temporal_offsets_curr_frames.append(-t_2)
                else:
                    temporal_offsets_curr_frames.append(t_2)
            temporal_frames = curr_frame + torch.tensor(temporal_offsets_curr_frames, device=reference_points.device)

            if reference_points_per_frame.shape[-1] == 4:
                reference_points_distance =  reference_points_per_frame[temporal_frames, :, :2] - reference_points_per_frame[curr_frame, :, :2][None]
            else:
                reference_points_distance =  reference_points_per_frame[temporal_frames] - reference_points_per_frame[curr_frame][None]

            reference_points_distance = reference_points_distance.transpose(0, 1)
            total_offset_distance = temporal_offsets_per_frame[curr_frame] + reference_points_distance / distance_normalizer[None]
            total_offsets.append(total_offset_distance)

            if dec_embedding_correlation_alignment and reference_points_per_frame.shape[-1] == 4:
                temporal_alignments = reference_points_per_frame[temporal_frames, :, 2:] / distance_normalizer
                total_alignment.append(temporal_alignments.transpose(0, 1))
                # total_alignment.append(torch.stack(frame_alignments, dim=1))

        total_offsets = torch.cat(total_offsets, dim=0)
        total_offsets = total_offsets[:, None, :, :] * valid_ratios[0, None, :, None]

        if total_alignment and dec_embedding_correlation_alignment:
            total_alignment = torch.cat(total_alignment, dim=0)
            total_alignment = total_alignment[:, None, :, :] * valid_ratios[0, None, :, None]
            return total_offsets, total_alignment

        return total_offsets


    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, targets=None):

        output = tgt
        intermediate = []
        intermediate_reference_points = []

        if self.dec_connect_all_embeddings:
            embds_per_frame = reference_points.shape[1] // self.num_frames
            if self.sort_temporal_offsets:
                curr_temporal_offset, other_temporal_offsets, temporal_positions_per_frame = self.get_curr_frame_and_temporal_offsets_decoder_all_connect_sorted(src_spatial_shapes, embds_per_frame, src_valid_ratios, tgt.device)

            else:
                curr_temporal_offset, other_temporal_offsets = self.get_curr_frame_and_temporal_offsets_decoder_all_connect_parallel(src_spatial_shapes,
                                                                                                                    embds_per_frame,
                                                                                                                    src_valid_ratios, tgt.device)
        else:
            curr_temporal_offset, other_temporal_offsets = self.get_curr_frame_and_temporal_offsets_decoder(src_spatial_shapes, reference_points.shape[1], self.temporal_window, src_valid_ratios, tgt.device)

        targets["other_temporal_offsets"] = other_temporal_offsets
        targets["curr_temporal_offset"] = curr_temporal_offset
        targets["src_spatial_shapes"] = src_spatial_shapes
        targets[f"ref_point_{0}"] = reference_points

        other_frames_temporal_offsets =  torch.clone(other_temporal_offsets)

        for lid, layer in enumerate(self.layers):
            targets["layer"] = lid
            if self.dec_embedding_correlation:
                if self.dec_connect_all_embeddings:
                    if self.sort_temporal_offsets:
                        other_frames_temporal_offsets = self.get_temporal_offset_from_same_instance_all_connect_sorted_offsets_parallel(reference_points, other_temporal_offsets, temporal_positions_per_frame, src_spatial_shapes, src_valid_ratios, self.dec_embedding_correlation_alignment)

                    else:
                        other_frames_temporal_offsets = self.get_temporal_offset_from_same_instance_all_connect_parallel(reference_points, other_temporal_offsets, src_spatial_shapes, src_valid_ratios, self.dec_embedding_correlation_alignment)

                else:
                    other_frames_temporal_offsets = self.get_temporal_offset_from_same_instance_parallel(reference_points, self.temporal_window,
                                                                                                               other_temporal_offsets, src_spatial_shapes,
                                                                                                               src_valid_ratios, self.dec_embedding_correlation_alignment)
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios[0, None], src_valid_ratios[0, None]], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[0, None]

            output, targets = layer(output, query_pos, reference_points_input, curr_temporal_offset, other_frames_temporal_offsets, src, src_spatial_shapes, src_level_start_index, src_padding_mask, targets)

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

            targets[f"ref_point_{lid+1}"] = reference_points

            if self.ref_point_embed is not None:
                tmp = self.ref_point_embed[lid](output)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                reference_points = new_reference_points.sigmoid()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)


        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), targets

        return output, reference_points, targets


def build_deforamble_transformerVIZ(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        num_frames=args.num_frames,
        nhead_enc=args.nheads_enc,
        nhead_dec=args.nheads_dec,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        enc_temporal_window= args.enc_temporal_window,
        dec_temporal_window=args.dec_temporal_window,
        dec_embedding_correlation= args.with_embedding_correlation,
        dec_embedding_correlation_alignment = args.with_embedding_correlation_alignment,
        dec_n_curr_points=args.dec_n_curr_points,
        dec_n_temporal_points=args.dec_n_temporal_points,
        dec_connect_all_embeddings=args.dec_connect_all_embeddings,
        enc_n_curr_points=args.enc_n_curr_points,
        enc_n_temporal_points=args.enc_n_temporal_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        use_trajectory_queries=args.use_trajectory_queries,
        with_gradient=args.with_gradient,
        with_decoder_instance_self_attn=args.with_decoder_instance_self_attn,
        with_decoder_frame_self_attn=args.with_decoder_frame_self_attn,
        enc_connect_all_embeddings=args.enc_connect_all_embeddings,
        dec_sort_temporal_offsets=args.dec_sort_temporal_offsets,
        enc_use_new_sampling_init_default=args.enc_use_new_sampling_init_default,

    )

