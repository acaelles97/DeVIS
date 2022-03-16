# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .deformable_transformer import build_deforamble_transformer
from .deformable_transformer_newclass import build_deforamble_transformer_new_class
from .original_def_encoder_transformer import build_NonTemporal_deformable_transformer
from .original_def_decoder_transformerr import build_deforamble_non_temporal_decoder_transformer
from .deformable_transformer_att_map_viz import build_deforamble_transformerVIZ
from .deformable_transformer_unified import build_deforamble_unified_transformer

from .original_transformer import build_original_transformer
from .deformable_vistr import DeformableVisTR, AllPostProcessor
from .deformable_vistr_newclass import DeformableVisTRNewClass
from .deformable_segmentation import DeformableVisTRsegm
from .deformable_segmentation_2 import DeformableVisTRsegm2
from .deformable_segmentation_3 import DeformableVisTRsegm3
from .deformable_segmentation_4 import DeformableVisTRsegm4
from .deformable_segmentation_definitive import DeformableVisTRsegmDefinitive
from .deformable_segmentation_final import DeformableVisTRsegmFinal
from .vistr import VisTR, SetCriterion
from .segmentation import VisTRsegm, PostProcessSegm

def build_model(num_classes, args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    matcher = build_matcher(args)

    post_processor_kwargs = {
        'focal_loss': args.focal_loss,
        'num_frames': args.num_frames,
        'top_k_inference': args.top_k_inference,
        'use_instance_level_classes': args.use_instance_level_classes,
    }

    postprocessors = AllPostProcessor(**post_processor_kwargs)

    vistr_kwargs = {
        'backbone': backbone,
        'num_classes': num_classes - 1 if args.focal_loss and not args.softmax_activation else num_classes,
        'num_queries': args.num_queries,
        'num_frames': args.num_frames,
        'aux_loss': args.aux_loss, }

    if args.deformable:
        if args.non_temporal_encoder and args.non_temporal_decoder:
            transformer = build_original_transformer(args)

        elif args.non_temporal_encoder:
            transformer = build_NonTemporal_deformable_transformer(args)

        elif args.non_temporal_decoder:
            assert not args.non_temporal_encoder
            transformer = build_deforamble_non_temporal_decoder_transformer(args)

        elif args.new_temporal_connection:
            transformer = build_deforamble_unified_transformer(args)

        elif args.viz_att_maps:
            transformer = build_deforamble_transformerVIZ(args)

        else:
            transformer = build_deforamble_transformer(args)

        vistr_kwargs['transformer'] = transformer
        vistr_kwargs['num_feature_levels'] = args.num_feature_levels
        vistr_kwargs['with_box_refine'] = args.with_box_refine
        vistr_kwargs["two_stage"] = args.two_stage
        vistr_kwargs["use_trajectory_queries"] = args.use_trajectory_queries
        vistr_kwargs["with_ref_point_refine"] = args.with_ref_point_refine
        vistr_kwargs["with_gradient"] = args.with_gradient
        vistr_kwargs["with_single_class_embed"] = args.with_single_class_embed
        vistr_kwargs["with_class_inst_attn"] = args.with_class_inst_attn

        model = DeformableVisTR(**vistr_kwargs)

        if args.masks:
            mask_kwargs = {
                'only_positive_matches':  args.only_positive_matches,
                'matcher': matcher,
                'use_deformable_conv': args.use_deformable_conv,
                'mask_head_used_features':  args.mask_head_used_features,
                'att_maps_used_res': args.att_maps_used_res,
                'post_processor':  postprocessors,
                'top_k_inference': args.top_k_inference,
                'vistr': model,
                'mask_aux_loss': args.mask_aux_loss,
                # 'save_matcher_viz': args.only_positive_matches and args.matcher_save_viz,
            }
            if args.new_segm_module:
                if  args.new_segm_module == "default":
                    model = DeformableVisTRsegm2(**mask_kwargs)

                elif args.new_segm_module == "mask_attention":
                    mask_kwargs['mask_attn_alignment'] = args.mask_attn_alignment
                    mask_kwargs['use_box_coords'] = args.use_box_coords
                    model = DeformableVisTRsegm3(**mask_kwargs)

                elif args.new_segm_module == "mask_heatmap":
                    mask_kwargs['use_ct_distance'] = args.use_ct_distance
                    model = DeformableVisTRsegm4(**mask_kwargs)

                elif args.new_segm_module == "final":
                    mask_kwargs["mask_head_out_mdcn"] = args.mask_head_out_mdcn
                    model = DeformableVisTRsegmFinal(**mask_kwargs)

                elif args.new_segm_module == "definitive":
                    model = DeformableVisTRsegmDefinitive(**mask_kwargs)

            else:
                model = DeformableVisTRsegm(**mask_kwargs)

    else:
        transformer = build_transformer(args)

        vistr_kwargs['transformer'] = transformer
        model = VisTR(**vistr_kwargs)

        if args.masks:
            mask_kwargs = {
                'vistr': model
            }
            model = VisTRsegm(**mask_kwargs)

    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef, }

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        if args.balance_aux_loss:
            layer_weighting_dict = {
                5: 1/2,
                4: 5/30,
                3: 4/30,
                2: 3/30,
                1: 2/30,
                0: 1/30,
            }
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v * layer_weighting_dict[i] for k, v in weight_dict.items()})
            weight_dict['loss_ce'] *= layer_weighting_dict[5]
            weight_dict['loss_bbox'] *= layer_weighting_dict[5]
            weight_dict['loss_giou'] *= layer_weighting_dict[5]
            weight_dict.update(aux_weight_dict)

        else:
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses.append('masks')
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
        if args.mask_aux_loss:
            assert isinstance(args.mask_aux_loss, list), "args.mask_aux_loss not a list"
            assert len(args.mask_aux_loss) == len(set(args.mask_aux_loss)), f"Use unique levels number in args.mask_aux_loss, Value {args.mask_aux_loss}"
            assert  min(args.mask_aux_loss) >= 0 and max(args.mask_aux_loss) <= 4, f"Available aux_loss levels : [0, 1, 2, 3, 4], Value {args.mask_aux_loss}"

            layer_weighting_mask_dict = {
                3: 2 / 3,
                1: 1 / 3,
            }
            if args.balance_mask_aux_loss:
                for i in args.mask_aux_loss:
                    weight_dict[f"loss_mask_{i}"] = args.mask_loss_coef * layer_weighting_mask_dict[i]
                    weight_dict[f"loss_dice_{i}"] = args.dice_loss_coef * layer_weighting_mask_dict[i]
            else:
                for i in args.mask_aux_loss:
                    weight_dict[f"loss_mask_{i}"] = args.mask_loss_coef
                    weight_dict[f"loss_dice_{i}"] = args.dice_loss_coef

        if args.new_segm_module == "final":
            losses.append("centroid")
            weight_dict["loss_centroids"] = args.centroid_loss_coef
            if args.balance_mask_aux_loss:
                for i in args.mask_aux_loss:
                    weight_dict[f"loss_centroids_{i}"] = args.centroid_loss_coef  * layer_weighting_mask_dict[i]

            else:
                if args.mask_aux_loss:
                    for i in args.mask_aux_loss:
                        weight_dict[f"loss_centroids_{i}"] = args.centroid_loss_coef


    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        num_frames=args.num_frames,
        use_instance_level_classes = args.use_instance_level_classes,
        volumetric_mask_loss=args.volumetric_mask_loss,
        with_class_no_obj=args.with_class_no_obj,
        softmax_activation=args.softmax_activation,
        with_loss_vistr_policy=args.with_loss_vistr_policy)

    criterion.to(device)

    return model, criterion, postprocessors
