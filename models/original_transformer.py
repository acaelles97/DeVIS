# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from .original_def_decoder_transformerr import DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .original_def_encoder_transformer import DeformableTransformerEncoder, DeformableTransformerEncoderLayer
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
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

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, num_feature_levels, nhead_dec, dec_n_curr_points)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, with_gradient, return_intermediate_dec)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
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
        assert self.two_stage or query_embed is not None
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
        memory = self.encoder(src_flatten, spatial_shapes_enc, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = memory.shape
        if self.use_trajectory_queries:
            query_embed = query_embed.repeat(self.num_frames, 1)
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(1, -1, -1)
        tgt = tgt.unsqueeze(0).expand(1, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points


        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes_enc, level_start_index, valid_ratios, query_embed, mask_flatten)


        spatial_shapes_mask_head = torch.cat((torch.tensor([[self.num_frames]], device=spatial_shapes_enc.device).repeat(self.num_lvls, 1), spatial_shapes_enc), dim=1)
        level_start_index_mask_head = torch.cat((spatial_shapes_mask_head.new_zeros((1,)), spatial_shapes_mask_head.prod(1).cumsum(0)[:-1]))

        inter_references_out = inter_references

        offset = 0
        memory_slices = []
        batch_size, _, channels = memory.shape
        for src in srcs:
            _, _, height, width = src.shape
            memory_slice = memory[:, offset:offset + height * width].permute(0, 2, 1).view(batch_size, channels, height, width)
            memory_slices.append(memory_slice)
            offset += height * width

        mem_flatten = []
        for mem in memory_slices:
            n, c, h, w = mem.shape
            src = mem.reshape(n // self.num_frames, self.num_frames, c, h, w).permute(0, 2, 1, 3, 4).flatten(-3)
            mem_flatten.append(src)

        mem_flatten = torch.cat(mem_flatten, 2).permute(0, 2, 1)

        return hs, query_embed, mem_flatten, init_reference_out, inter_references_out, level_start_index_mask_head, valid_ratios, spatial_shapes_mask_head, None


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_original_transformer(args):
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
    )
