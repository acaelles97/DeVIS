import torch


def add_decoder_bbox_embed_to_weights(inst_segm_weights, bbox_refinement_weights):
    inst_segm_model = torch.load(inst_segm_weights, map_location='cpu')['model']
    bbox_refinement_model = torch.load(bbox_refinement_weights, map_location='cpu')['model']
    print("A")
    for key, value in bbox_refinement_model.items():
        if 'decoder.bbox_embed' in key:
            assert key not in inst_segm_model.keys()
            inst_segm_model[key] = value

    return inst_segm_model


def add_r101_weights(inst_segm_weights, bbox_refinement_weights):
    inst_segm_model = torch.load(inst_segm_weights, map_location='cpu')['model']
    backbone_weights = torch.load(bbox_refinement_weights, map_location='cpu')['model']
    new101_weights = {}

    for key, value in inst_segm_model.items():
        if 'backbone' not in key:
            new101_weights[key] = value

    for key, value in backbone_weights.items():
        if 'backbone' in key:
            new101_weights[".".join(key.split(".")[1:])] = value

    return new101_weights


if __name__ == "__main__":
    instance_segm_weights = "/usr/stud/cad/projects/VisTR/weights/r50_deformable_detr_DEFAULT_instance_segm_plus_iterative_bbox_refinement.pth"
    iterative_bbox_refinment_weights = "/usr/stud/cad/projects/VisTR/weights/r101_weight.pth"
    out_name = "/usr/stud/cad/projects/VisTR/weights/r101_deformable_detr_DEFAULT_instance_segm_plus_iterative_bbox_refinement.pth"

    new_weights = add_r101_weights(instance_segm_weights, iterative_bbox_refinment_weights)
    new_weights = {"model": new_weights}
    torch.save(new_weights, out_name)
