import json
import os
from typing import List, Dict
import numpy as np
import torch
from torch.nn.init import kaiming_uniform_, normal_

def read_json(dataset_json_path: str):
    with open(dataset_json_path, 'r') as fh:
        dataset = json.load(fh)
    return dataset


def save_json(dataset: Dict, dataset_json_path: str):
    print(f"Saving to {dataset_json_path}")
    with open(dataset_json_path, 'w') as file:
        json.dump(dataset, file)


def parse_categories_by_name(yvis_cats, ovis_cats):
    conversion_dict = {cat["name"]: cat["id"] for cat in yvis_cats}
    ovis_cats = {cat["name"]: cat["id"] for cat in ovis_cats}
    matched_cats = set()
    unmatched_cats = set()
    neuron_ids = np.zeros(len(ovis_cats)+1) - 1
    for cat, cat_id in ovis_cats.items():
        cat_name = cat.lower()
        if cat_name in conversion_dict.keys():
            matched_cats.add(cat)
            neuron_ids[cat_id] = conversion_dict[cat_name]
        else:
            unmatched_cats.add(cat)

    # Manually add Car neuron for Vehical and Bird for Parrot
    neuron_ids[ovis_cats["Motorcycle"]] = conversion_dict["motorbike"]
    neuron_ids[ovis_cats["Vehical"]] = conversion_dict["sedan"]

    # Background knowledge
    neuron_ids[0] = 0

    return neuron_ids

def parse_categories_by_name_for_coco(coco_cats, ovis_cats):
    conversion_dict = {cat["name"]: cat["id"] for cat in coco_cats}
    ovis_cats = {cat["name"]: cat["id"] for cat in ovis_cats}
    matched_cats = set()
    unmatched_cats = set()
    neuron_ids = np.zeros(len(ovis_cats)+1) - 1
    for cat, cat_id in ovis_cats.items():
        cat_name = cat.lower()
        if cat_name in conversion_dict.keys():
            matched_cats.add(cat)
            neuron_ids[cat_id] = conversion_dict[cat_name]
        else:
            unmatched_cats.add(cat)

    # Manually add Car neuron for Vehical and Bird for Parrot
    neuron_ids[ovis_cats["Vehical"]] = conversion_dict["car"]
    neuron_ids[ovis_cats["Parrot"]] = conversion_dict["bird"]

    return neuron_ids


def parse_categories_by_name_for_yvis_from_coco(coco_cats, yvis_cats):
    conversion_dict = {cat["name"]: cat["id"] for cat in coco_cats}
    yvis_cats = {cat["name"]: cat["id"] for cat in yvis_cats}
    matched_cats = set()
    unmatched_cats = set()
    neuron_ids = np.zeros(len(yvis_cats)+1) - 1
    for cat, cat_id in yvis_cats.items():
        cat_name = cat.lower()
        if cat_name in conversion_dict.keys():
            matched_cats.add(cat)
            neuron_ids[cat_id-1] = conversion_dict[cat_name]
        else:
            unmatched_cats.add(cat)

    # Manually add Car neuron for Vehical and Bird for Parrot
    neuron_ids[yvis_cats["motorbike"]] = conversion_dict["motorcycle"]
    neuron_ids[yvis_cats["sedan"]] = conversion_dict["car"]

    return neuron_ids



def convert_weights_dict(checkpoint, neurons_ids):
    inidices_to_gather = neurons_ids != -1
    logits_to_gather = neurons_ids[inidices_to_gather]

    class_embed_weight = checkpoint["vistr.class_embed.weight"]
    new_class_embed_weight = kaiming_uniform_(torch.Tensor(len(neurons_ids), class_embed_weight.shape[1]))
    new_class_embed_weight[inidices_to_gather] = class_embed_weight[logits_to_gather]
    checkpoint["vistr.class_embed.weight"] = new_class_embed_weight


    class_embed_bias = checkpoint["vistr.class_embed.bias"]
    new_class_embed_bias = kaiming_uniform_(torch.Tensor(len(neurons_ids), 1))[:, 0]
    new_class_embed_bias[inidices_to_gather] = class_embed_bias[logits_to_gather]
    checkpoint["vistr.class_embed.bias"] = new_class_embed_bias

    query_embed_weight = checkpoint["vistr.query_embed.weight"]

    indxes = [False] * 25
    indxes[:10] = [True] * 10
    indxes = indxes * 36
    new_query_embed_weight = normal_(torch.Tensor(36 * 24, query_embed_weight.shape[1]))
    new_query_embed_weight[indxes] = query_embed_weight
    checkpoint["vistr.query_embed.weight"] = new_query_embed_weight

def convert_weights_dict_from_detr(detr_checkpoint, neurons_ids):
    inidices_to_gather = neurons_ids != -1
    logits_to_gather = neurons_ids[inidices_to_gather]

    class_embed_weight = detr_checkpoint["vistr.class_embed.weight"]
    new_class_embed_weight = kaiming_uniform_(torch.Tensor(len(neurons_ids), class_embed_weight.shape[1]))
    new_class_embed_weight[inidices_to_gather] = class_embed_weight[logits_to_gather]
    detr_checkpoint["vistr.class_embed.weight"] = new_class_embed_weight


    class_embed_bias = detr_checkpoint["vistr.class_embed.bias"]
    new_class_embed_bias = kaiming_uniform_(torch.Tensor(len(neurons_ids), 1))[:, 0]
    new_class_embed_bias[inidices_to_gather] = class_embed_bias[logits_to_gather]
    detr_checkpoint["vistr.class_embed.bias"] = new_class_embed_bias



if __name__ == "__main__":
    yvis_cats_path = "/usr/stud/cad/p028/data/Youtube_VIS/train/train.json"
    coco_cats_path = "/usr/stud/cad/p028/data/COCO/annotations/instances_val2017.json"

    # ovis_cats_path = "/usr/prakt/p028/data/OVIS/original_ovis_train.json"
    # output_path = "/usr/prakt/p028/data/Annotations/mapping_logits_coco_ovis.npy"
    # weights_path = "/usr/prakt/p028/projects/VisTR/weights/vistr_r50.pth"
    # detr_weights_path = "/usr/prakt/p028/projects/VisTR/weights/384_coco_r50.pth"
    # out_detr_weihts_path = "/usr/prakt/p028/projects/VisTR/weights/detr_for_ovis_weights.pth"
    #
    # out_weights_path = "/usr/prakt/p028/projects/VisTR/weights/vistr_r50_36_24_ovis_transformed.pth"


    # checkpoint = torch.load(weights_path, map_location='cpu')['model']
    # coco_cats = read_json(yvis_cats_path)["categories"]
    # ovis_cats = read_json(ovis_cats_path)["categories"]
    # neuron_ids = parse_categories_by_name(coco_cats, ovis_cats)
    # convert_weights_dict(checkpoint, neuron_ids)
    # torch.save(checkpoint, out_weights_path)

    # checkpoint = torch.load(detr_weights_path, map_location='cpu')['model']
    coco_cats = read_json(coco_cats_path)["categories"]
    yvis_cats = read_json(yvis_cats_path)["categories"]
    neuron_ids = parse_categories_by_name_for_yvis_from_coco(coco_cats, yvis_cats)

    # convert_weights_dict_from_detr(checkpoint, neuron_ids)
    # torch.save(checkpoint, out_detr_weihts_path)