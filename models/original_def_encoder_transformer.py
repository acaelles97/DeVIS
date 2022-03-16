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
from .ops.modules import MSDeformAttn, TemporalDifferentModuleMSDeformAttn
from .transformer import _get_clones, _get_activation_fn
from .deformable_transformer import DeformableTransformerDecoderLayer, DeformableTransformerDecoder

class NonTemporalDeformableTransformer(nn.Module):
    def __init__(self, d_model=256, num_frames=6, nhead_enc=8, nhead_dec=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, enc_temporal_window=2, dec_temporal_window=2, dec_connect_all_embeddings=False, dec_embedding_correlation=False, dec_embedding_correlation_alignment=False,
                 dec_n_curr_points=4,  enc_n_curr_points=4, dec_n_temporal_points=2, enc_n_temporal_points=2,
                 two_stage=False, two_stage_num_proposals=300, use_trajectory_queries=False, with_gradient=False, with_decoder_instance_self_attn=False,
                 with_decoder_frame_self_attn=False, enc_connect_all_embeddings=False, dec_sort_temporal_offsets=False):

        super().__init__()

        self.d_model = d_model
        self.num_lvls = num_feature_levels
        self.use_trajectory_queries = use_trajectory_queries
        self.num_frames = num_frames
        self.nhead_enc = nhead_enc
        self.nhead_dec = nhead_dec
        self.with_gradient = with_gradient

        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, num_frames, enc_temporal_window,
                                                          num_feature_levels, nhead_enc, enc_n_curr_points, enc_n_temporal_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        if dec_connect_all_embeddings:
            assert dec_embedding_correlation
            dec_temporal_window = self.num_frames - 1

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
            if isinstance(m, (MSDeformAttn, TemporalDifferentModuleMSDeformAttn)):
                m._reset_parameters()
        if not self.two_stage:
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
        # prepare input for encoder
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed[0].flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes_enc = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes_enc.new_zeros((1,)), spatial_shapes_enc.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes_enc, level_start_index, valid_ratios, lvl_pos_embed_flatten, padding_mask=None)

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape
        for src in srcs:
            _, _, height, width = src.shape
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(batch_size, channels, height, width)
            memory_slices.append(memory_slice)
            offset += height * width

        mem_flatten = []
        for lvl, mem in enumerate(memory_slices):
            n, c, h, w = mem.shape
            mem_temp = mem.reshape(n // self.num_frames, self.num_frames, c, h, w).permute(0, 2, 1, 3, 4).flatten(-3)
            mem_flatten.append(mem_temp)

        mem_flatten = torch.cat(mem_flatten, 2).permute(0, 2, 1)

        spatial_shapes_dec = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        spatial_shapes_dec = torch.cat((torch.tensor([[self.num_frames]], device=spatial_shapes_dec.device).repeat(self.num_lvls, 1), spatial_shapes_dec), dim=1)
        level_start_index_dec = torch.cat((spatial_shapes_dec.new_zeros((1,)), spatial_shapes_dec.prod(1).cumsum(0)[:-1]))

        bs, _, c = mem_flatten.shape
        if self.use_trajectory_queries:
            query_embed = query_embed.repeat(self.num_frames, 1)
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(
            tgt, reference_points, mem_flatten, spatial_shapes_dec, level_start_index_dec, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        return hs, query_embed, mem_flatten, init_reference_out, inter_references_out, level_start_index_dec, valid_ratios, spatial_shapes_dec






class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_frames=6, t_window=2,  n_levels=4, n_heads=8, n_curr_points=4, n_temporal_points=2):

        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_curr_points)
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

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 =  self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output



def build_NonTemporal_deformable_transformer(args):
    return NonTemporalDeformableTransformer(
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
    )