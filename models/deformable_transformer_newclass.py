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
import torch.nn.functional as F
from .deformable_transformer import DeformableTransformerEncoderLayer, DeformableTransformerEncoder

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, num_frames=6, nhead_enc=8, nhead_dec=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, enc_temporal_window=2, dec_temporal_window=2, dec_connect_all_embeddings=False, dec_embedding_correlation=False, dec_embedding_correlation_alignment=False,
                 dec_n_curr_points=4,  enc_n_curr_points=4, dec_n_temporal_points=2, enc_n_temporal_points=2,
                 two_stage=False, two_stage_num_proposals=300, use_trajectory_queries=False, class_head_type="class_weight"):

        super().__init__()

        self.d_model = d_model
        self.num_lvls = num_feature_levels
        self.use_trajectory_queries = use_trajectory_queries
        self.num_frames = num_frames
        self.nhead_enc = nhead_enc
        self.nhead_dec = nhead_dec

        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, num_frames, enc_temporal_window,
                                                          num_feature_levels, nhead_enc, enc_n_curr_points, enc_n_temporal_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, enc_temporal_window)

        if dec_connect_all_embeddings:
            assert dec_embedding_correlation
            dec_temporal_window = self.num_frames - 1

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation, num_frames, dec_temporal_window,
                                                          num_feature_levels, nhead_dec, dec_n_curr_points, dec_n_temporal_points, dec_embedding_correlation, class_head_type)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, dec_embedding_correlation, dec_embedding_correlation_alignment, dec_connect_all_embeddings, num_frames,  dec_temporal_window, return_intermediate_dec)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
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
        hs, hs_classes, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index,
            valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        return hs, hs_classes, query_embed, memory, init_reference_out, inter_references_out, level_start_index, valid_ratios, spatial_shapes





class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_frames=36, t_window=2, n_levels=4, n_heads=8, n_curr_points=4,
                 n_temporal_points=2, dec_embedding_correlation=False, class_head_type="class_weight"):

        super().__init__()
        self.cross_attn = TemporalDifferentModuleMSDeformAttn(n_frames, d_model, n_levels, t_window, n_heads, n_curr_points, n_temporal_points,
                                                              True, dec_embedding_correlation, class_head_type == "class_weight")

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
        self.time_attention_weights = None
        if class_head_type == "class_weight":
            self.time_attention_weights = nn.Linear(d_model, 1)

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
        tgt2, tgt_class = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, curr_temporal_offset,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        if self.time_attention_weights is not None:
            num_frames = src_spatial_shapes[0][0]
            time_weight = self.time_attention_weights(tgt)
            time_weight = time_weight.reshape(tgt.shape[0], num_frames, time_weight.shape[1] // num_frames, 1)
            time_weight = F.softmax(time_weight, 1)

            tgt_class = tgt_class.reshape(tgt.shape[0], num_frames, tgt.shape[1] // num_frames, tgt_class.shape[-1])
            tgt_class = (tgt_class * time_weight).sum(1)
            return tgt, tgt_class

        else:
            return tgt, None

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, dec_embedding_correlation, dec_embedding_correlation_alignment, dec_connect_all_embeddings, num_frames, temporal_window, return_intermediate=False):
        super().__init__()
        self.dec_connect_all_embeddings = dec_connect_all_embeddings
        self.dec_embedding_correlation_alignment = dec_embedding_correlation_alignment
        self.dec_embedding_correlation = dec_embedding_correlation
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

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
                        frame_offsets.append(torch.tensor([0, -t_1 / T_], device=device))

                    # Padding scenario 1) Pad with the offset referring to last frame of sequence
                    elif t_1 + t_2 > T_ - 1:
                        frame_offsets.append(torch.tensor([0, (T_ - 1 - t_1) / T_], device=device))

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
    def get_temporal_offset_from_same_instance2(reference_points, temporal_offsets, input_spatial_shapes, temporal_window, valid_ratios,
                                                dec_embedding_correlation_alignment, device):
        T_ = input_spatial_shapes[0][0]
        embds_per_frame = reference_points.shape[1] // T_

        temporal_offsets_per_frame = temporal_offsets[:, 0].reshape(T_, embds_per_frame, temporal_offsets.shape[-2], temporal_offsets.shape[-1])
        reference_points_per_frame = reference_points.reshape(T_, embds_per_frame, reference_points.shape[-1])
        total_offsets = []
        total_alignment = []
        distance_normalizer = torch.tensor([[1, input_spatial_shapes[0][0]]], device=input_spatial_shapes.device)

        # We will have 3 different scenarios
        # 1) First frame of the clip: We will sample past frame as future frames reversed
        temporal_frames_first = [t for t in range((temporal_window // 2) + 1) if t != 0]
        temporal_frames_first = temporal_frames_first[::-1] + temporal_frames_first

        # 2) Last frame of the clip: We will sample future frames as past frames reversed
        temporal_frames_last = [-t for t in range((temporal_window // 2) + 1) if t != 0]
        temporal_frames_last = temporal_frames_last[::-1] + temporal_frames_last

        # 3) In-between frames: Follow the sliding window approach across each one
        temporal_frames_in_between = [t for t in range(-temporal_window // 2, (temporal_window // 2) + 1) if t != 0]

        for t_1 in range(0, T_):
            frame_offsets = []
            frame_alignments = []

            if t_1 == 0 or t_1 == T_ - 1:
                frames_to_iterate = temporal_frames_first if t_1 == 0 else temporal_frames_last
                # Special case first frame
                for pos, t_2 in enumerate(frames_to_iterate):
                    if reference_points_per_frame.shape[-1] == 4:
                        reference_points_distance = reference_points_per_frame[t_1 + t_2, :, :2] - reference_points_per_frame[t_1, :, :2]
                        if dec_embedding_correlation_alignment:
                            frame_alignments.append(reference_points_per_frame[t_1 + t_2, :, 2:] / distance_normalizer)
                    else:
                        assert reference_points.shape[-1] == 2
                        reference_points_distance = reference_points_per_frame[t_1 + t_2] - reference_points_per_frame[t_1]

                    total_offset_distance = temporal_offsets_per_frame[t_1, :, pos] + reference_points_distance / distance_normalizer
                    frame_offsets.append(total_offset_distance)

            else:
                for pos, t_2 in enumerate(temporal_frames_in_between):
                    # Padding scenario 1) Pad with the offset referring to first frame of sequence
                    if t_1 + t_2 < 0:
                        timestep_used = 0

                    # Padding scenario 1) Pad with the offset referring to last frame of sequence
                    elif t_1 + t_2 > T_ - 1:
                        timestep_used = -1

                    else:
                        timestep_used = t_1 + t_2

                    if reference_points_per_frame.shape[-1] == 4:
                        reference_points_distance = reference_points_per_frame[timestep_used, :, :2] - reference_points_per_frame[t_1, :, :2]
                        if dec_embedding_correlation_alignment:
                            frame_alignments.append(reference_points_per_frame[t_1 + t_2, :, 2:] / distance_normalizer)
                    else:
                        assert reference_points.shape[-1] == 2
                        reference_points_distance = reference_points_per_frame[timestep_used] - reference_points_per_frame[t_1]

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


    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):

        output = tgt
        intermediate = []
        intermediate_reference_points = []
        intermediate_class = []

        if self.dec_connect_all_embeddings:
            curr_temporal_offset, other_temporal_offsets = self.get_curr_frame_and_temporal_offsets_decoder_all_connect(src_spatial_shapes, reference_points.shape[1], self.temporal_window, src_valid_ratios, tgt.device)
        else:
            curr_temporal_offset, other_temporal_offsets = self.get_curr_frame_and_temporal_offsets_decoder(src_spatial_shapes, reference_points.shape[1], self.temporal_window, src_valid_ratios, tgt.device)

        other_frames_temporal_offsets =  torch.clone(other_temporal_offsets)

        # Compute distances that will remain the same for all the layers
        if self.bbox_embed is None and self.dec_embedding_correlation:
            # Reference points will remain static so we dont need to compute temporal offsets each layer
            if self.dec_connect_all_embeddings:
                other_frames_temporal_offsets = self.get_temporal_offset_from_same_instance_all_connect(reference_points, other_temporal_offsets,
                                                                                                        src_spatial_shapes, self.temporal_window,
                                                                                                        src_valid_ratios, tgt.device)
            else:
                other_frames_temporal_offsets = self.get_temporal_offset_from_same_instance2(reference_points, other_temporal_offsets,
                                                                                                       src_spatial_shapes, self.temporal_window,
                                                                                                       src_valid_ratios, self.dec_embedding_correlation_alignment, tgt.device)

        for lid, layer in enumerate(self.layers):
            if self.dec_embedding_correlation and self.bbox_embed is not None:
                if self.dec_connect_all_embeddings:
                    other_frames_temporal_offsets = self.get_temporal_offset_from_same_instance_all_connect(reference_points, other_temporal_offsets, src_spatial_shapes, self.temporal_window, src_valid_ratios, self.dec_embedding_correlation_alignment, tgt.device)
                else:
                    other_frames_temporal_offsets = self.get_temporal_offset_from_same_instance2(reference_points,
                                                                                                               other_temporal_offsets, src_spatial_shapes, self.temporal_window,
                                                                                                               src_valid_ratios, self.dec_embedding_correlation_alignment, tgt.device)
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios[0, None], src_valid_ratios[0, None]], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[0, None]


            output, output_class = layer(output, query_pos, reference_points_input, curr_temporal_offset, other_frames_temporal_offsets, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

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
                reference_points = new_reference_points.detach()

            if self.ref_point_embed is not None:
                tmp = self.ref_point_embed[lid](output)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                reference_points = new_reference_points.sigmoid()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_class.append(output_class)
                intermediate_reference_points.append(reference_points)


        if self.return_intermediate:
            if intermediate_class[0] is None:
                return torch.stack(intermediate), None, torch.stack(intermediate_reference_points)
            else:
                return torch.stack(intermediate), torch.stack(intermediate_class), torch.stack(intermediate_reference_points)

        return output, reference_points


def build_deforamble_transformer_new_class(args):
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
        class_head_type= args.class_head_type,
    )