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
from .deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder

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
        self.with_gradient = with_gradient
        self.num_frames = num_frames
        self.nhead_enc = nhead_enc
        self.nhead_dec = nhead_dec

        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        if enc_connect_all_embeddings:
            enc_temporal_window = self.num_frames - 1

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, num_frames, enc_temporal_window,
                                                          num_feature_levels, nhead_enc, enc_n_curr_points, enc_n_temporal_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, enc_temporal_window, enc_connect_all_embeddings)

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
            if isinstance(m, (MSDeformAttn, TemporalDifferentModuleMSDeformAttn)):
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
        spatial_shapes_dec = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        spatial_shapes_enc = torch.cat((torch.tensor([[self.num_frames]], device=spatial_shapes_dec.device).repeat(self.num_lvls, 1), spatial_shapes_dec), dim=1)

        level_start_index_enc = torch.cat((spatial_shapes_enc.new_zeros((1,)), spatial_shapes_enc.prod(1).cumsum(0)[:-1]))
        level_start_index_dec = torch.cat((spatial_shapes_dec.new_zeros((1,)), spatial_shapes_dec.prod(1).cumsum(0)[:-1]))

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes_enc, level_start_index_enc, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # unflatten memory
        offset = 0
        memory_slices = []
        for src in srcs:
            n, c, h, w = src.shape
            memory_slice = memory[:, offset:offset + h * w * self.num_frames].permute(0, 2, 1).reshape(n // self.num_frames, c, self.num_frames, h, w)
            memory_slice = memory_slice[0].flatten(-2).permute(1,2,0)
            memory_slices.append(memory_slice)
            offset += h * w * self.num_frames

        memory_per_frame = torch.cat(memory_slices, dim=1)

        # prepare input for decoder
        bs, _, c = memory.shape

        if self.use_trajectory_queries:
            query_embed = query_embed.repeat(self.num_frames, 1)
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(
            tgt, reference_points, memory_per_frame, spatial_shapes_dec, level_start_index_dec, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        return hs, query_embed, memory, init_reference_out, inter_references_out, level_start_index_enc, valid_ratios, spatial_shapes_enc


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
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

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        if tgt.shape[0] != query_pos.shape[0]:
            tgt = tgt.flatten(0, 1)[None]

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt = tgt.reshape([src.shape[0], tgt.shape[1] // src.shape[0], tgt.shape[-1]])
        query_embed_dec = query_pos.reshape([src.shape[0], query_pos.shape[1] // src.shape[0], query_pos.shape[-1]])

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_embed_dec), reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, with_gradient, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        self.with_gradient = with_gradient
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    # tgt ->  Embeddings parameter from the bbx per se / reference_points -> initial guess for each embedding of a coordinate (x,y) / memory -> output from transformer encoder / spatial_shapes -> intitial spatial resolution of each of the features maps that we have obtained from the the backbone and then feed to the trans. encoder / query_pos -> other chunk of learnable parameters

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios, query_pos=None, src_padding_mask=None):
        output = tgt

        reference_points = reference_points.reshape([src.shape[0], reference_points.shape[1] // src.shape[0], reference_points.shape[-1]])

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                # Adds the level axis to reference_points_input
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

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

            if self.return_intermediate:
                intermediate.append(output.flatten(0, 1)[None])
                intermediate_reference_points.append(reference_points.flatten(0, 1)[None])

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points



def build_deforamble_non_temporal_decoder_transformer(args):
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