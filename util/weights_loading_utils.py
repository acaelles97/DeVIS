import numpy as np
import torch


def adapt_weights_vistr(checkpoint, model_state_dict, eval_only):
    if not eval_only:
        del checkpoint["vistr.class_embed.weight"]
        del checkpoint["vistr.class_embed.bias"]
        del checkpoint["vistr.query_embed.weight"]

    checkpoint_state_dict = {}

    for k, v in checkpoint.items():
        if k.startswith("detr"):
            checkpoint_state_dict[k.replace("detr","vistr")] = v
        else:
            checkpoint_state_dict[k] = v

    resume_state_dict = {}
    for k, v in model_state_dict.items():
        if k not in checkpoint_state_dict:
            print(f'Load {k} {tuple(v.shape)} from scratch.')

        resume_state_dict[k] = v

    return resume_state_dict


def adapt_weights_evis(checkpoint, model_state_dict, lvl_res, focal_loss, load_class_neurons, num_frames, finetune_query_embds,
                       finetune_attn_mask_head,  trajectory_queries, with_decoder_instance_self_attn, finetune_temporal_modules, enc_temporal_window, enc_n_temporal_points, dec_n_temporal_points):

    IDS_COCO_TO_YVIS = np.array(
        [1, -1, -1, -1, 41, -1, 3, 18, -1, -1, -1, -1, -1, 17, 21, -1, 7, 19, -1, 23, -1, 4, -1, -1, -1, -1, 42, 5, 8, 24,
         -1, 22, 36, 9, -1, 74, -1, -1, -1, -1, -1])

    checkpoint_state_dict = {}

    for k, v in checkpoint['model'].items():
        if k.startswith("detr"):
            checkpoint_state_dict[k.replace('detr', 'vistr')] = v
        elif "vistr" not in k and ("transformer" in k or "class_embed" in k or "bbox_embed" in k or "input_proj" in k or "query_embed" in k or "backbone" in k):
            checkpoint_state_dict[f"vistr.{k}"] = v
            if finetune_temporal_modules and "transformer.encoder" in k and "self_attn" in k and not 'value_proj' in k and not 'output_proj' in k:
                name = "vistr." +  ".".join(k.split(".")[:5]) + ".temporal_"+ ".".join(k.split(".")[5:])
                checkpoint_state_dict[name] = torch.clone(v)

            if finetune_temporal_modules and "transformer.decoder" in k and "cross_attn" in k and not 'value_proj' in k and not 'output_proj' in k:
                name = "vistr." +  ".".join(k.split(".")[:5]) + ".temporal_"+ ".".join(k.split(".")[5:])
                checkpoint_state_dict[name] = torch.clone(v)

            if lvl_res == 1 and "input_proj.2" in  k:
                name = "vistr." + k.split(".")[0] + ".0."+".".join(k.split(".")[2:])
                checkpoint_state_dict[name] = torch.clone(v)

            if finetune_attn_mask_head and "transformer.decoder.layers.5" in k:
                name = ".".join(k.split(".")[4:])
                checkpoint_state_dict[f"mask_embd_attention.module.{name}"] = torch.clone(v)

            if with_decoder_instance_self_attn and "transformer.decoder.layers" in k and ("self_attn" in k or "norm2" in k):
                if "self_attn" in k:
                    name = "vistr." + ".".join(k.split(".")[:4]) + ".self_attn_inst." + ".".join(k.split(".")[5:])
                else:
                    name =  "vistr." +".".join(k.split(".")[:4]) + ".norm2_inst." + ".".join(k.split(".")[5:])

                checkpoint_state_dict[name] = torch.clone(v)

        elif k.startswith("cross_attn"):
            checkpoint_state_dict[f"mask_embd_attention.module.{k}"] = v

        else:
            checkpoint_state_dict[k] = v

    resume_state_dict = {}
    window_size = num_frames - 1
    # window_size = 6
    for k, v in model_state_dict.items():
        if k not in checkpoint_state_dict or ('query_embed' in k and not finetune_query_embds) or ('class_embed' in k and not load_class_neurons):
            resume_value = v
            print(f'Load {k} {tuple(v.shape)} from scratch.')

        elif 'query_embed' in k and finetune_query_embds and ((v.shape[0] // num_frames == checkpoint_state_dict[k].shape[0] and not trajectory_queries) or (trajectory_queries and v.shape[0] == checkpoint_state_dict[k].shape[0])):
            raise NotImplementedError
            # print(f'Load {k} {tuple(v.shape)} from MODEL')
            # if trajectory_queries:
            #     resume_value = checkpoint_state_dict[k]
            # else:
            #     resume_value = checkpoint_state_dict[k].repeat(num_frames, 1)

        elif finetune_temporal_modules and 'self_attn' in k and 'temporal' in k and 'transformer.encoder' in k:
            checkpoint_value = checkpoint_state_dict[k]
            print(f"Adapting value {k}")
            if 'sampling_offsets' in  k:
                if 'bias' not in k:
                    sampling_offsets = checkpoint_value.view(8, 4, 1, 4, 2, 256).repeat(1, 1, window_size, 1, 1, 1)
                    resume_value = sampling_offsets[:, :lvl_res, :, :enc_n_temporal_points].reshape(-1, 256)
                else:
                    sampling_offsets = checkpoint_value.view(8, 4, 1, 4, 2).repeat(1, 1, window_size, 1, 1)
                    resume_value = sampling_offsets[:, :lvl_res, :, :enc_n_temporal_points].reshape(-1)
            else:
                if 'bias' not in k:
                    att_weights = checkpoint_value.view(8, 4, 1, 4, 256).repeat(1, 1, window_size, 1, 1)
                    resume_value = att_weights[:, :lvl_res, :, :enc_n_temporal_points].reshape(-1, 256)

                else:
                    att_weights = checkpoint_value.view(8, 4, 1, 4).repeat(1, 1, window_size, 1)
                    resume_value = att_weights[:, :lvl_res, :, :enc_n_temporal_points].reshape(-1)


        elif finetune_temporal_modules and 'cross_attn' in k and 'temporal' in k and 'transformer.decoder' in k:
            checkpoint_value = checkpoint_state_dict[k]
            print(f"Adapting value {k}")
            if 'sampling_offsets' in k:
                if 'bias' not in k:
                    sampling_offsets = checkpoint_value.view(8, 4, 1, 4, 2, 256).repeat(1, 1, window_size, 1, 1, 1)
                    resume_value = sampling_offsets[:, :lvl_res, :, :dec_n_temporal_points].reshape(-1, 256)
                else:
                    sampling_offsets = checkpoint_value.view(8, 4, 1, 4, 2).repeat(1, 1, window_size, 1, 1)
                    resume_value = sampling_offsets[:, :lvl_res, :, :dec_n_temporal_points].reshape(-1)
            else:
                if 'bias' not in k:
                    att_weights = checkpoint_value.view(8, 4, 1, 4, 256).repeat(1, 1, window_size, 1, 1)
                    resume_value = att_weights[:, :lvl_res, :, :dec_n_temporal_points].reshape(-1, 256)

                else:
                    att_weights = checkpoint_value.view(8, 4, 1, 4).repeat(1, 1, window_size, 1)
                    resume_value = att_weights[:, :lvl_res, :, :dec_n_temporal_points].reshape(-1)



        elif k in checkpoint_state_dict and v.shape != checkpoint_state_dict[k].shape:
            print(f"Adapting value {k}")
            checkpoint_value = checkpoint_state_dict[k]
            if 'level_embed' in k:
                resume_value = checkpoint_value[:v.shape[0]]

            # elif 'module.cross_attn.sampling_offsets' in k:
            #     if 'bias' not in k:
            #         sampling_offsets = checkpoint_value.view(8, 4, 4, 2, 256)
            #         resume_value = sampling_offsets[:, :lvl_res, :2].reshape(-1, 256)
            #
            #     else:
            #         sampling_offsets = checkpoint_value.view(8, 4, 4, 2)
            #         resume_value = sampling_offsets[:, :lvl_res, :2].reshape(-1)
            #
            # elif 'module.cross_attn.attention_weights' in k:
            #     if 'bias' not in k:
            #         att_weights = checkpoint_value.view(8, 4, 4, 256)
            #         resume_value = att_weights[:, :lvl_res, :2].reshape(-1, 256)
            #     else:
            #         att_weights = checkpoint_value.view(8, 4, 4)
            #         resume_value = att_weights[:, :lvl_res, :2].reshape(-1)

            elif 'self_attn.attention_weights' in k or 'cross_attn.attention_weights' in k:
                if 'bias' not in k:
                    att_weights = checkpoint_value.view(8, 4, 4, 256)
                    resume_value = att_weights[:, :lvl_res].reshape(-1, 256)
                else:
                    att_weights = checkpoint_value.view(8, 4, 4)
                    resume_value = att_weights[:, :lvl_res].reshape(-1)

            elif 'self_attn.sampling_offsets' in k or 'cross_attn.sampling_offsets' in k:
                if 'bias' not in k:
                    sampling_offsets = checkpoint_value.view(8, 4, 4, 2, 256)
                    resume_value = sampling_offsets[:, :lvl_res].reshape(-1, 256)
                else:
                    sampling_offsets = checkpoint_value.view(8, 4, 4, 2)
                    resume_value = sampling_offsets[:, :lvl_res].reshape(-1)
            elif 'class_embed' in k:
                assert load_class_neurons
                if focal_loss:
                    IDS_COCO_TO_YVIS_ = IDS_COCO_TO_YVIS[:-1]
                else:
                    IDS_COCO_TO_YVIS_ = IDS_COCO_TO_YVIS

                inidices_to_gather = IDS_COCO_TO_YVIS_ != -1
                logits_to_gather = IDS_COCO_TO_YVIS_[inidices_to_gather] - 1
                tmp_v = v.clone().to('cpu')
                tmp_v[inidices_to_gather] = checkpoint_value[logits_to_gather]
                resume_value = tmp_v


            elif 'query_embed' in k and finetune_query_embds:
                if trajectory_queries:
                    picked_percentage = checkpoint_value.shape[0] // v.shape[0]
                    picked_idxs = []
                    for i in range(checkpoint_value.shape[0]):
                        if i % picked_percentage == 0:
                            picked_idxs.append(i)
                    embd_idx = torch.tensor(picked_idxs)[:v.shape[0]]
                    resume_value = checkpoint_value[embd_idx]

                else:
                    picked_idxs = []
                    num_queries_to_gather = v.shape[0] // num_frames
                    picked_percentage = checkpoint_value.shape[0] // num_queries_to_gather
                    for i in range(checkpoint_value.shape[0]):
                        if i % picked_percentage == 0:
                            picked_idxs.append(i)
                    embd_idx = torch.tensor(picked_idxs)[:v.shape[0]]
                    resume_value = checkpoint_value[embd_idx].repeat(num_frames, 1)

            #
            #     elif query_embed_init == "finetune_wo_alignment":
            #         v[:checkpoint_value.shape[0]] = checkpoint_value
            #         resume_value = v
            #
            #     elif query_embed_init == "finetune_with_alignment":
            #         checkpoint_pos_embd, checkpoint_tgt_embd = torch.split(checkpoint_value, checkpoint_value.shape[1] // 2, dim=1)
            #         _, tgt_embd  = torch.split(v, checkpoint_value.shape[1] // 2, dim=1)
            #
            #         # Fill tgt embeddings with the ones we can finetune
            #         tgt_embd[:checkpoint_value.shape[0]] = checkpoint_tgt_embd
            #
            #         #Pick num_traj pos_embd so trajectories start with different embeddings but same reference_point
            #         num_trajectories = v.shape[0] // num_frames
            #
            #         traj_pos_embd = checkpoint_pos_embd[:num_trajectories].repeat(num_frames, 1, 1).flatten(0, 1).to(tgt_embd.device)
            #
            #         resume_value = torch.cat([traj_pos_embd, tgt_embd], dim=1)
            #     else:
            #         resume_value = v

            else:
                raise NotImplementedError
        else:
            resume_value = checkpoint_state_dict[k]

        resume_state_dict[k] = resume_value

    for key in checkpoint_state_dict.keys():
        if key not in resume_state_dict.keys():
            print(f"Ignoring {key} from checkpoint")


    return resume_state_dict

def adapt_weights_unified_model(checkpoint, model_state_dict, lvl_res, focal_loss, load_class_neurons, num_frames, finetune_query_embds, trajectory_queries, enc_n_points, dec_n_points):

    IDS_COCO_TO_YVIS = np.array(
        [1, -1, -1, -1, 41, -1, 3, 18, -1, -1, -1, -1, -1, 17, 21, -1, 7, 19, -1, 23, -1, 4, -1, -1, -1, -1, 42, 5, 8, 24,
         -1, 22, 36, 9, -1, 74, -1, -1, -1, -1, -1])

    checkpoint_state_dict = {}

    for k, v in checkpoint['model'].items():
        if k.startswith("detr"):
            checkpoint_state_dict[k.replace('detr', 'vistr')] = v
        elif "vistr" not in k and ("transformer" in k or "class_embed" in k or "bbox_embed" in k or "input_proj" in k or "query_embed" in k or "backbone" in k):
            checkpoint_state_dict[f"vistr.{k}"] = v

        else:
            checkpoint_state_dict[k] = v

    resume_state_dict = {}
    for k, v in model_state_dict.items():
        if k not in checkpoint_state_dict or ('query_embed' in k and not finetune_query_embds) or ('class_embed' in k and not load_class_neurons):
            resume_value = v
            print(f'Load {k} {tuple(v.shape)} from scratch.')

        elif 'query_embed' in k and finetune_query_embds and ((v.shape[0] // num_frames == checkpoint_state_dict[k].shape[0] and not trajectory_queries) or (trajectory_queries and v.shape[0] == checkpoint_state_dict[k].shape[0])):
            print(f'Load {k} {tuple(v.shape)} from MODEL')
            if trajectory_queries:
                resume_value = checkpoint_state_dict[k]
            else:
                resume_value = checkpoint_state_dict[k].repeat(num_frames, 1)

        elif k in checkpoint_state_dict and v.shape != checkpoint_state_dict[k].shape:
            print(f"Adapting value {k}")
            checkpoint_value = checkpoint_state_dict[k]
            if 'level_embed' in k:
                resume_value = checkpoint_value[:v.shape[0]]

            elif 'encoder' in  k and 'self_attn.attention_weights' in k:
                if 'bias' not in k:
                    # att_weights1 = checkpoint_value.transpose(0, 1).view(256, 8, 4, 1, 4).repeat(1, 1, 1, num_frames, 1)
                    # resume_value2 = att_weights1[:, :, :lvl_res, :, :enc_n_points].reshape(256, -1).transpose(0, 1)
                    att_weights = checkpoint_value.view(8, 4, 1, 4, 256).repeat(1, 1, num_frames, 1, 1)
                    resume_value = att_weights[:, :lvl_res, :, :enc_n_points].reshape(-1, 256)

                else:
                    att_weights = checkpoint_value.view(8, 4, 1, 4).repeat(1, 1, num_frames, 1)
                    resume_value = att_weights[:, :lvl_res, :, :enc_n_points].reshape(-1)

            elif 'encoder' in k and 'self_attn.sampling_offsets' in k:
                if 'bias' not in k:
                    sampling_offsets = checkpoint_value.view(8, 4, 1, 4, 2, 256).repeat(1, 1, num_frames, 1, 1, 1)
                    resume_value = sampling_offsets[:, :lvl_res, :, :enc_n_points].reshape(-1, 256)

                else:
                    sampling_offsets = checkpoint_value.view(8, 4, 1, 4, 2).repeat(1, 1, num_frames, 1, 1)
                    resume_value = sampling_offsets[:, :lvl_res, :, :enc_n_points].reshape(-1)


            elif 'decoder' in k and 'cross_attn.attention_weights' in k:
                if 'bias' not in k:
                    att_weights = checkpoint_value.view(8, 4, 1, 4, 256).repeat(1, 1, num_frames, 1, 1)
                    resume_value = att_weights[:, :lvl_res, :, :dec_n_points].reshape(-1, 256)

                else:
                    att_weights = checkpoint_value.view(8, 4, 1, 4).repeat(1, 1, num_frames, 1)
                    resume_value = att_weights[:, :lvl_res, :, :dec_n_points].reshape(-1)


            elif 'decoder' in k and 'cross_attn.sampling_offsets' in k:
                if 'bias' not in k:
                    sampling_offsets = checkpoint_value.view(8, 4, 1, 4, 2, 256).repeat(1, 1, num_frames, 1, 1, 1)
                    resume_value = sampling_offsets[:, :lvl_res, :, :dec_n_points].reshape(-1, 256)
                else:
                    sampling_offsets = checkpoint_value.view(8, 4, 1, 4, 2).repeat(1, 1, num_frames, 1, 1)
                    resume_value = sampling_offsets[:, :lvl_res, :, :dec_n_points].reshape(-1)

            elif 'class_embed' in k:
                assert load_class_neurons
                if focal_loss:
                    IDS_COCO_TO_YVIS_ = IDS_COCO_TO_YVIS[:-1]
                else:
                    IDS_COCO_TO_YVIS_ = IDS_COCO_TO_YVIS

                inidices_to_gather = IDS_COCO_TO_YVIS_ != -1
                logits_to_gather = IDS_COCO_TO_YVIS_[inidices_to_gather] - 1
                tmp_v = v.clone().to('cpu')
                tmp_v[inidices_to_gather] = checkpoint_value[logits_to_gather]
                resume_value = tmp_v


            elif 'query_embed' in k and finetune_query_embds:
                picked_percentage = checkpoint_value.shape[0] // v.shape[0]
                picked_idxs = []
                for i in range(checkpoint_value.shape[0]):
                    if i % picked_percentage == 0:
                        picked_idxs.append(i)
                embd_idx = torch.tensor(picked_idxs)[:v.shape[0]]
                resume_value = checkpoint_value[embd_idx]

            else:
                raise NotImplementedError
        else:
            resume_value = checkpoint_state_dict[k]

        resume_state_dict[k] = resume_value

    for key in checkpoint_state_dict.keys():
        if key not in resume_state_dict.keys():
            print(f"Ignoring {key} from checkpoint")


    return resume_state_dict


def add_decoder_bbox_embed_to_weights(inst_segm_weights, bbox_refinement_weights):
    inst_segm_model =  torch.load(inst_segm_weights, map_location='cpu')['model']
    bbox_refinement_model = torch.load(bbox_refinement_weights, map_location='cpu')['model']
    print("A")
    for key, value in bbox_refinement_model.items():
        if 'decoder.bbox_embed' in key:
            assert key not in inst_segm_model.keys()
            inst_segm_model[key] = value

    return inst_segm_model

def add_r101_weights(inst_segm_weights, bbox_refinement_weights):
    inst_segm_model =  torch.load(inst_segm_weights, map_location='cpu')['model']
    backbone_weights = torch.load(bbox_refinement_weights, map_location='cpu')['model']
    print("A")
    new101_weights = {}

    for key, value in inst_segm_model.items():
        if 'backbone'  not in key:
            new101_weights[key] = value

    for key, value in backbone_weights.items():
        if 'backbone'  in key:
            new101_weights[".".join(key.split(".")[1:])] = value


    return new101_weights


if __name__ == "__main__":
    instance_segm_weights = "/usr/stud/cad/projects/VisTR/weights/r50_deformable_detr_DEFAULT_instance_segm_plus_iterative_bbox_refinement.pth"
    iterative_bbox_refinment_weights = "/usr/stud/cad/projects/VisTR/weights/r101_weight.pth"
    out_name = "/usr/stud/cad/projects/VisTR/weights/r101_deformable_detr_DEFAULT_instance_segm_plus_iterative_bbox_refinement.pth"

    new_weights = add_r101_weights(instance_segm_weights, iterative_bbox_refinment_weights)
    new_weights = {"model": new_weights}
    torch.save(new_weights, out_name)