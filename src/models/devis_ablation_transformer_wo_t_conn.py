# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from .deformable_transformer import DeformableTransformer, DeformableTransformerEncoderLayer, \
    DeformableTransformerEncoder, DeformableTransformerDecoderLayer, \
    DeformableTransformerDecoder
from .devis_transformer import DeVISTransformer


class DeVISAblationTransformer(DeVISTransformer):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, activation="relu", num_feature_levels=4,
                 enc_n_points=4, dec_n_points=4, with_gradient=False):
        # These are the only arguments that build modules that will not be replaced, so we can
        # ommit the rest ones
        super().__init__(d_model=d_model, nhead=nhead, num_feature_levels=num_feature_levels,
                         with_gradient=with_gradient)

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeVISAblationTransformerDecoderLayer(d_model, dim_feedforward,
                                                             dropout, activation,
                                                             num_feature_levels, nhead,
                                                             dec_n_points)

        self.decoder = DeVISAblationTransformerDecoder(decoder_layer, num_decoder_layers,
                                                       with_gradient)

        self._reset_parameters()

    # def forward(self, srcs, masks, pos_embeds, query_embed=None):
    #
    #     src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes, level_start_index, valid_ratios = self.prepare_data(srcs, masks, pos_embeds)
    #
    #     # encoder
    #     memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
    #
    #     # prepare input for decoder
    #     T_, _, channels = memory.shape
    #
    #     query_embed, tgt = torch.split(query_embed, channels, dim=1)
    #     query_embed = query_embed.unsqueeze(0)
    #     tgt = tgt.unsqueeze(0)
    #     reference_points = self.reference_points(query_embed).sigmoid()
    #     init_reference_out = reference_points
    #
    #     # decoder
    #     hs, inter_references = self.decoder(tgt, reference_points, memory, spatial_shapes, level_start_index,
    #                                         valid_ratios, query_embed)
    #
    #     inter_references_out = inter_references
    #
    #     offset = 0
    #     memories = []
    #     for src in srcs:
    #         _, _, height, width = src.shape
    #         memory_slice = memory[:, offset:offset + height * width].permute(2, 0, 1).view(1, channels, T_, height, width)
    #         memories.append(memory_slice)
    #         offset += height * width
    #
    #     return hs, query_embed, memories, init_reference_out, inter_references_out, level_start_index, valid_ratios, spatial_shapes


# class DeVISTransformerEncoderLayer(DeformableTransformerEncoderLayer):
#     def __init__(self, d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_frames=6, t_window=2, n_levels=4, n_heads=8, n_curr_points=4, n_temporal_points=2):
#         super().__init__(d_model=d_model, d_ffn=d_ffn, dropout=dropout, activation=activation, n_levels=n_levels, n_heads=n_heads, n_points=None)
#
#         self.self_attn = TemporalMSDeformAttnEncoder(n_frames, d_model, n_levels, t_window, n_heads, n_curr_points, n_temporal_points)
#

# class DeVISTransformerEncoder(DeformableTransformerEncoder):
#     def __init__(self, encoder_layer, num_layers, t_window, enc_connect_all_embeddings):
#         super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)
#         self.t_window = t_window
#         self.enc_connect_all_embeddings = enc_connect_all_embeddings
#
#     def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
#         output = src
#         reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
#
#         T_ = src.shape[0]
#         temporal_offsets = []
#         if self.enc_connect_all_embeddings:
#             temporal_spatial_shapes = spatial_shapes.repeat(T_ - 1, 1)
#             for curr_frame in range(0, T_):
#                 frame_offsets = torch.tensor([t for t in range(-curr_frame, T_ - curr_frame) if t != 0], device=src.device)
#                 temporal_offsets.append(frame_offsets)
#
#         else:
#             temporal_spatial_shapes = spatial_shapes.repeat(self.t_window, 1)
#             temporal_frames = [t for t in range(-self.t_window // 2, (self.t_window // 2) + 1) if t != 0]
#             for curr_frame in range(0, T_):
#                 frame_offsets = []
#                 for t_frame in temporal_frames:
#                     if curr_frame + t_frame < 0 or curr_frame + t_frame > T_ - 1:
#                         frame_offsets.append(-t_frame)
#                     else:
#                         frame_offsets.append(t_frame)
#                 temporal_offsets.append(torch.tensor(frame_offsets, device=src.device))
#
#         kwargs = {
#             "temporal_offsets": temporal_offsets
#         }
#
#         temporal_level_start_index = torch.cat((temporal_spatial_shapes.new_zeros((1,)), temporal_spatial_shapes.prod(1).cumsum(0)[:-1]))
#
#         for _, layer in enumerate(self.layers):
#             output = layer(output, pos, reference_points, (spatial_shapes, temporal_spatial_shapes), (level_start_index, temporal_level_start_index), **kwargs)
#
#         return output


class DeVISAblationTransformerDecoderLayer(DeformableTransformerDecoderLayer):

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                **kwargs):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                              tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt = tgt.view([src.shape[0], tgt.shape[1] // src.shape[0], tgt.shape[-1]])
        query_embed_dec = query_pos.view(
            [src.shape[0], query_pos.shape[1] // src.shape[0], query_pos.shape[-1]])
        reference_points = reference_points.view((src.shape[0],
                                                  reference_points.shape[1] // src.shape[0]) +
                                                 reference_points.shape[-2:])

        tgt2, *_ = self.cross_attn(self.with_pos_embed(tgt, query_embed_dec), reference_points, src,
                               src_spatial_shapes, level_start_index, **kwargs)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        tgt = tgt.flatten(0, 1)[None]

        return tgt


class DeVISAblationTransformerDecoder(DeformableTransformerDecoder):
    def __init__(self, decoder_layer, num_layers, with_gradient=False):
        super().__init__(decoder_layer=decoder_layer, num_layers=num_layers,
                         with_gradient=with_gradient)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
                src_valid_ratios,
                query_pos=None, src_padding_mask=None):

        output = tgt
        intermediate = []
        intermediate_reference_points = []

        kwargs = {
            'input_padding_mask': src_padding_mask
        }

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat(
                    [src_valid_ratios[0, None], src_valid_ratios[0, None]], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[0, None]

            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes,
                           src_level_start_index, **kwargs)

            reference_points, intermediate, intermediate_reference_points = \
                self.refine_reference_point(lid, output, reference_points, intermediate,
                                            intermediate_reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


def build_devis_ablation_transformer(cfg):
    return DeVISAblationTransformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD,
        dropout=cfg.MODEL.DROPOUT,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        with_gradient=cfg.MODEL.BBX_GRADIENT_PROP,

        num_encoder_layers=cfg.MODEL.TRANSFORMER.ENCODER_LAYERS,
        num_decoder_layers=cfg.MODEL.TRANSFORMER.DECODER_LAYERS,
        nhead=cfg.MODEL.TRANSFORMER.N_HEADS,
        enc_n_points=cfg.MODEL.TRANSFORMER.ENC_N_POINTS,
        dec_n_points=cfg.MODEL.TRANSFORMER.DEC_N_POINTS,

        activation="relu",

    )
