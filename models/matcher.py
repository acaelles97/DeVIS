"""
Instance Sequence Matching
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, multi_iou, multi_giou

INF = 100000000
from util.mask_ops import compute_iou_matrix
import numpy as np
import pycocotools.mask as mask_util
from typing import List

class SoftMaxHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_frames: int = 36, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, num_queries: int = 360,
                 focal_loss: bool = False, focal_alpha: float = 0.25, use_giou: bool = False, use_l1_distance_sum: bool = False, use_trajectory_queries:bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        self.num_out = num_queries if use_trajectory_queries else num_queries // num_frames
        self.focal_loss = False
        self.use_giou = use_giou
        self.use_l1_distance_sum = use_l1_distance_sum
        self.focal_alpha = focal_alpha
        self.use_trajectory_queries = use_trajectory_queries
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, return_embd_idx=False):
        """ Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        valid_ordered_ind = []

        original_valid_ordered_ind = []

        for i in range(bs):

            out_prob = outputs["pred_logits"][i].softmax(-1)
            num_classes = out_prob.shape[-1] - 1
            out_bbox = outputs["pred_boxes"][i]
            tgt_ids = torch.clone(targets[i]["labels"])
            tgt_bbox = targets[i]["boxes"]
            tgt_valid = targets[i]["valid"]
            tgt_original_valid = targets[i]["original_valid"]
            num_tgt = len(tgt_ids) // self.num_frames
            if num_tgt == 0:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device),
                                torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
                if return_embd_idx:
                    valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]
                    original_valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]
                    return indices, None, valid_ordered_ind, original_valid_ordered_ind

                return indices

            gamma = 2.0

            tgt_valid_split = tgt_valid.reshape(num_tgt, self.num_frames)
            tgt_original_valid_split = tgt_original_valid.reshape(num_tgt, self.num_frames)
            tgt_ids = tgt_ids.reshape(num_tgt, self.num_frames)
            out_bbox_split = out_bbox.reshape(self.num_frames, self.num_out, out_bbox.shape[-1]).permute(1, 0, 2).unsqueeze(1)
            tgt_bbox_split = tgt_bbox.reshape(num_tgt, self.num_frames, 4).unsqueeze(0)
            out_prob = out_prob.reshape(self.num_frames, self.num_out, out_prob.shape[-1]).permute(1, 0, 2)

            tgt_ids_background = tgt_original_valid_split == 0
            tgt_ids[tgt_ids_background] = num_classes
            total_class_loss, total_bbox_cost, total_iou_cost = [], [], []

            for tgt_idx in range(tgt_valid_split.shape[0]):
                tgt_valid_instance = tgt_valid_split[tgt_idx].type(torch.bool)
                if not torch.any(tgt_valid_instance):
                    continue
                tgt_ids_instance = tgt_ids[tgt_idx][tgt_valid_instance].long()
                valid_frame = torch.tensor([i for i in range(self.num_frames) if tgt_valid_instance[i]], device=tgt_valid.device).long()
                num_frames = torch.sum(tgt_valid_instance).item()
                cost_class_instance = -out_prob[:, valid_frame, tgt_ids_instance]
                total_class_loss.append(cost_class_instance.view(self.num_out, 1, num_frames).mean(dim=-1))

                tgt_bbx_instance = tgt_bbox_split[:, tgt_idx, valid_frame].unsqueeze(0)
                out_bbox_instance = out_bbox_split[:, :, valid_frame]

                if self.use_l1_distance_sum:
                    bbx_l1_distance =  torch.cdist(out_bbox_instance[:, 0].transpose(1, 0), tgt_bbx_instance[:, 0].transpose(1, 0), p=1)
                    bbx_l1_distance = bbx_l1_distance.mean(0)
                else:
                    bbx_l1_distance =  (out_bbox_instance - tgt_bbx_instance).abs().mean((-1, -2))

                total_bbox_cost.append(bbx_l1_distance)

                if self.use_giou:
                    iou = multi_giou(box_cxcywh_to_xyxy(out_bbox_instance), box_cxcywh_to_xyxy(tgt_bbx_instance)).mean(-1)
                else:
                    iou = multi_iou(box_cxcywh_to_xyxy(out_bbox_instance), box_cxcywh_to_xyxy(tgt_bbx_instance)).mean(-1)

                total_iou_cost.append(-1 * iou)

            if not total_class_loss:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device),
                                torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
                if return_embd_idx:
                    valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]
                    original_valid_ordered_ind =  [torch.tensor([]).long().to(out_prob.device)]
                    return indices, None, valid_ordered_ind, original_valid_ordered_ind
                return indices

            class_cost = torch.cat(total_class_loss, dim=1)
            bbox_cost = torch.cat(total_bbox_cost, dim=1)
            iou_cost =  torch.cat(total_iou_cost, dim=1)

            # TODO: only deal with box and mask with empty target
            cost = self.cost_class * class_cost + self.cost_bbox * bbox_cost + self.cost_giou * iou_cost
            out_i, tgt_i = linear_sum_assignment(cost.cpu())

            index_i, index_j, index_tmp_i, index_tmp_j  = [], [], [], []
            for j in range(len(out_i)):
                tgt_valid_ind_j = tgt_valid_split[tgt_i[j]].nonzero().flatten()
                tgt_original_valid_split_j = tgt_original_valid_split[tgt_i[j]].nonzero().flatten()
                index_i.append(tgt_valid_ind_j * self.num_out + out_i[j])
                index_j.append(tgt_valid_ind_j + tgt_i[j] * self.num_frames)


                index_tmp_i.append(tgt_original_valid_split_j * self.num_out + out_i[j])
                index_tmp_j.append(tgt_original_valid_split_j + tgt_i[j] * self.num_frames)


            if index_i == [] or index_j == []:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device),
                                torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
            else:
                index_i = torch.cat(index_i).long()
                index_j = torch.cat(index_j).long()

                index_tmp_i = torch.cat(index_tmp_i).long()
                index_tmp_j = torch.cat(index_tmp_j).long()

                indices.append((index_i, index_j, index_tmp_i, index_tmp_j))

            if return_embd_idx:
                all_idxs = []
                for j in range(len(out_i)):
                    all_idxs.append(torch.arange(self.num_frames) * self.num_out + out_i[j])
                    valid_ordered_ind.append(tgt_valid_split[tgt_i[j]].bool())
                    original_valid_ordered_ind.append(tgt_original_valid_split[tgt_i[j]].bool())
                return indices, all_idxs, valid_ordered_ind, original_valid_ordered_ind
            else:
                return indices



class DeformableHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_frames: int = 36, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, num_queries: int = 360,
                 focal_loss: bool = False, focal_alpha: float = 0.25, use_giou: bool = False, use_l1_distance_sum: bool = False, use_trajectory_queries:bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        self.num_out = num_queries if use_trajectory_queries else num_queries // num_frames
        self.focal_loss = True
        self.use_giou = use_giou
        self.use_l1_distance_sum = use_l1_distance_sum
        self.focal_alpha = focal_alpha
        self.use_trajectory_queries = use_trajectory_queries
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, return_embd_idx=False):
        """ Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        valid_ordered_ind = []
        original_valid_ordered_ind = []
        for i in range(bs):

            out_prob = outputs["pred_logits"][i].sigmoid()
            out_bbox = outputs["pred_boxes"][i]
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_valid = targets[i]["valid"]
            tgt_original_valid = targets[i]["original_valid"]
            num_tgt = len(tgt_ids) // self.num_frames
            if num_tgt == 0:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device),
                                torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
                if return_embd_idx:
                    valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]
                    original_valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]

                    return indices, None, valid_ordered_ind, original_valid_ordered_ind
                return indices

            gamma = 2.0
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            neg_cost_class = neg_cost_class.reshape(self.num_frames, self.num_out, out_prob.shape[-1]).permute(1, 0, 2)

            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            pos_cost_class = pos_cost_class.reshape(self.num_frames, self.num_out, out_prob.shape[-1]).permute(1, 0, 2)

            cost_class = (pos_cost_class - neg_cost_class)
            tgt_valid_split = tgt_valid.reshape(num_tgt, self.num_frames)
            tgt_original_valid_split = tgt_original_valid.reshape(num_tgt, self.num_frames)
            tgt_ids = tgt_ids.reshape(num_tgt, self.num_frames)
            out_bbox_split = out_bbox.reshape(self.num_frames, self.num_out, out_bbox.shape[-1]).permute(1, 0, 2).unsqueeze(1)
            tgt_bbox_split = tgt_bbox.reshape(num_tgt, self.num_frames, 4).unsqueeze(0)

            total_class_loss, total_bbox_cost, total_iou_cost = [], [], []
            for tgt_idx in range(tgt_valid_split.shape[0]):
                tgt_valid_instance = tgt_valid_split[tgt_idx].type(torch.bool)
                if not torch.any(tgt_valid_instance):
                    continue
                tgt_ids_instance = tgt_ids[tgt_idx][tgt_valid_instance].long()
                valid_frame = torch.tensor([i for i in range(self.num_frames) if tgt_valid_instance[i]], device=tgt_valid.device).long()
                num_frames = torch.sum(tgt_valid_instance).item()
                cost_class_instance = cost_class[:, valid_frame, tgt_ids_instance]
                total_class_loss.append(cost_class_instance.view(self.num_out, 1, num_frames).mean(dim=-1))

                tgt_bbx_instance = tgt_bbox_split[:, tgt_idx, valid_frame].unsqueeze(0)
                out_bbox_instance = out_bbox_split[:, :, valid_frame]

                if self.use_l1_distance_sum:
                    bbx_l1_distance =  torch.cdist(out_bbox_instance[:, 0].transpose(1, 0), tgt_bbx_instance[:, 0].transpose(1, 0), p=1)
                    bbx_l1_distance = bbx_l1_distance.mean(0)
                else:
                    bbx_l1_distance =  (out_bbox_instance - tgt_bbx_instance).abs().mean((-1, -2))

                total_bbox_cost.append(bbx_l1_distance)

                if self.use_giou:
                    iou = multi_giou(box_cxcywh_to_xyxy(out_bbox_instance), box_cxcywh_to_xyxy(tgt_bbx_instance)).mean(-1)
                else:
                    iou = multi_iou(box_cxcywh_to_xyxy(out_bbox_instance), box_cxcywh_to_xyxy(tgt_bbx_instance)).mean(-1)

                total_iou_cost.append(-1 * iou)

            if not total_class_loss:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device),
                                torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
                if return_embd_idx:
                    valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]
                    original_valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]

                    return indices, None, valid_ordered_ind, original_valid_ordered_ind
                return indices

            class_cost = torch.cat(total_class_loss, dim=1)
            bbox_cost = torch.cat(total_bbox_cost, dim=1)
            iou_cost =  torch.cat(total_iou_cost, dim=1)

            # TODO: only deal with box and mask with empty target
            cost = self.cost_class * class_cost + self.cost_bbox * bbox_cost + self.cost_giou * iou_cost
            out_i, tgt_i = linear_sum_assignment(cost.cpu())

            index_i, index_j, index_tmp_i, index_tmp_j  = [], [], [], []
            for j in range(len(out_i)):
                tgt_valid_ind_j = tgt_valid_split[tgt_i[j]].nonzero().flatten()
                tgt_original_valid_split_j = tgt_original_valid_split[tgt_i[j]].nonzero().flatten()
                index_i.append(tgt_valid_ind_j * self.num_out + out_i[j])
                index_j.append(tgt_valid_ind_j + tgt_i[j] * self.num_frames)


                index_tmp_i.append(tgt_original_valid_split_j * self.num_out + out_i[j])
                index_tmp_j.append(tgt_original_valid_split_j + tgt_i[j] * self.num_frames)


            if index_i == [] or index_j == []:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device),
                                torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
            else:
                index_i = torch.cat(index_i).long()
                index_j = torch.cat(index_j).long()

                index_tmp_i = torch.cat(index_tmp_i).long()
                index_tmp_j = torch.cat(index_tmp_j).long()

                indices.append((index_i, index_j, index_tmp_i, index_tmp_j))

            if return_embd_idx:
                all_idxs = []
                for j in range(len(out_i)):
                    all_idxs.append(torch.arange(self.num_frames) * self.num_out + out_i[j])
                    valid_ordered_ind.append(tgt_valid_split[tgt_i[j]].bool())
                    original_valid_ordered_ind.append(tgt_original_valid_split[tgt_i[j]].bool())
                return indices, all_idxs, valid_ordered_ind, original_valid_ordered_ind
            else:
                return indices


class DeformableHungarianMatcherInstLevelClass(DeformableHungarianMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets, return_embd_idx=False):
        """ Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_boxes"].shape[:2]
        indices = []
        valid_ordered_ind = []

        for i in range(bs):
            out_prob = outputs["pred_logits"][i].sigmoid()
            out_bbox = outputs["pred_boxes"][i]
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_valid = targets[i]["valid"]
            num_tgt = len(tgt_bbox) // self.num_frames
            if num_tgt == 0:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
                if return_embd_idx:
                    valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]
                    return indices, None, valid_ordered_ind
                return indices

            gamma = 2.0
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            # neg_cost_class = neg_cost_class.reshape(self.num_frames, self.num_out, out_prob.shape[-1]).permute(1, 0, 2)

            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            # pos_cost_class = pos_cost_class.reshape(self.num_frames, self.num_out, out_prob.shape[-1]).permute(1, 0, 2)

            class_cost = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            tgt_valid_split = tgt_valid.reshape(num_tgt, self.num_frames)
            # tgt_ids = tgt_ids.reshape(num_tgt, self.num_frames)
            out_bbox_split = out_bbox.reshape(self.num_frames, self.num_out, out_bbox.shape[-1]).permute(1, 0, 2).unsqueeze(1)
            tgt_bbox_split = tgt_bbox.reshape(num_tgt, self.num_frames, 4).unsqueeze(0)

            total_class_loss, total_bbox_cost, total_iou_cost = [], [], []
            for tgt_idx in range(tgt_valid_split.shape[0]):
                tgt_valid_instance = tgt_valid_split[tgt_idx].type(torch.bool)
                if not torch.any(tgt_valid_instance):
                    continue
                # tgt_ids_instance = tgt_ids[tgt_idx][tgt_valid_instance].long()
                valid_frame = torch.tensor([i for i in range(self.num_frames) if tgt_valid_instance[i]], device=tgt_valid.device).long()
                # num_frames = torch.sum(tgt_valid_instance).item()
                # cost_class_instance = cost_class[:, valid_frame, tgt_ids_instance]
                # total_class_loss.append(cost_class_instance.view(self.num_out, 1, num_frames).mean(dim=-1))

                tgt_bbx_instance = tgt_bbox_split[:, tgt_idx, valid_frame].unsqueeze(0)
                out_bbox_instance = out_bbox_split[:, :, valid_frame]

                if self.use_l1_distance_sum:
                    bbx_l1_distance =  torch.cdist(out_bbox_instance[:, 0].transpose(1, 0), tgt_bbx_instance[:, 0].transpose(1, 0), p=1)
                    bbx_l1_distance = bbx_l1_distance.mean(0)
                else:
                    bbx_l1_distance =  (out_bbox_instance - tgt_bbx_instance).abs().mean((-1, -2))

                total_bbox_cost.append(bbx_l1_distance)

                if self.use_giou:
                    iou = multi_giou(box_cxcywh_to_xyxy(out_bbox_instance), box_cxcywh_to_xyxy(tgt_bbx_instance)).mean(-1)
                else:
                    iou = multi_iou(box_cxcywh_to_xyxy(out_bbox_instance), box_cxcywh_to_xyxy(tgt_bbx_instance)).mean(-1)

                total_iou_cost.append(-1 * iou)

            if not total_bbox_cost:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
                if return_embd_idx:
                    valid_ordered_ind = [torch.tensor([]).long().to(out_prob.device)]
                    return indices, None, valid_ordered_ind
                return indices

            # class_cost = torch.cat(total_class_loss, dim=1)
            bbox_cost = torch.cat(total_bbox_cost, dim=1)
            iou_cost =  torch.cat(total_iou_cost, dim=1)

            # TODO: only deal with box and mask with empty target
            cost = self.cost_class * class_cost + self.cost_bbox * bbox_cost + self.cost_giou * iou_cost
            out_i, tgt_i = linear_sum_assignment(cost.cpu())

            index_i, index_j = [], []
            for j in range(len(out_i)):
                tgt_valid_ind_j = tgt_valid_split[tgt_i[j]].nonzero().flatten()
                index_i.append(tgt_valid_ind_j * self.num_out + out_i[j])
                index_j.append(tgt_valid_ind_j + tgt_i[j] * self.num_frames)
            if index_i == [] or index_j == []:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
            else:
                index_i = torch.cat(index_i).long()
                index_j = torch.cat(index_j).long()
                indices.append((index_i, index_j))

            if return_embd_idx:
                all_idxs = []
                for j in range(len(out_i)):
                    all_idxs.append(torch.arange(self.num_frames) * self.num_out + out_i[j])
                    valid_ordered_ind.append(tgt_valid_split[tgt_i[j]].bool())
                return indices, all_idxs, valid_ordered_ind
            else:
                return indices


def build_matcher(args):
    if args.softmax_activation:
        return SoftMaxHungarianMatcher(num_frames=args.num_frames, cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                                          cost_giou=args.set_cost_giou, num_queries=args.num_queries, focal_loss=args.focal_loss,
                                          focal_alpha=args.focal_alpha, use_giou=args.use_giou, use_l1_distance_sum=args.use_l1_distance_sum,
                                          use_trajectory_queries=args.use_trajectory_queries)

    if args.use_instance_level_classes:
        return DeformableHungarianMatcherInstLevelClass(num_frames=args.num_frames, cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                                          cost_giou=args.set_cost_giou, num_queries=args.num_queries, focal_loss=args.focal_loss,
                                          focal_alpha=args.focal_alpha, use_giou=args.use_giou, use_l1_distance_sum=args.use_l1_distance_sum,
                                          use_trajectory_queries=args.use_trajectory_queries)

    if args.focal_loss:
        return DeformableHungarianMatcher(num_frames=args.num_frames, cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                                          cost_giou=args.set_cost_giou, num_queries=args.num_queries, focal_loss=args.focal_loss,
                                          focal_alpha=args.focal_alpha, use_giou=args.use_giou, use_l1_distance_sum=args.use_l1_distance_sum,
                                          use_trajectory_queries=args.use_trajectory_queries)
    else:
        return HungarianMatcher(num_frames=args.num_frames, cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox,
                                cost_giou=args.set_cost_giou, num_queries=args.num_queries)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_frames: int = 36, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, num_queries: int = 360):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_frames = num_frames
        self.num_out = num_queries // num_frames
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, return_embd_idx=False):
        """ Performs the sequence level matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        valid_ordered_ind = []
        for i in range(bs):
            out_prob = outputs["pred_logits"][i].softmax(-1)
            out_bbox = outputs["pred_boxes"][i]
            tgt_ids = targets[i]["labels"]
            tgt_bbox = targets[i]["boxes"]
            tgt_valid = targets[i]["valid"]
            num_tgt = len(tgt_ids) // self.num_frames
            if num_tgt == 0:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))
                if return_embd_idx:
                    valid_ordered_ind = torch.zeros(self.num_frames, dtype=torch.bool, device=out_prob.device)
                    return indices, None, valid_ordered_ind
                return indices

            out_prob_split = out_prob.reshape(self.num_frames, self.num_out, out_prob.shape[-1]).permute(1, 0, 2)
            out_bbox_split = out_bbox.reshape(self.num_frames, self.num_out, out_bbox.shape[-1]).permute(1, 0, 2).unsqueeze(1)
            tgt_bbox_split = tgt_bbox.reshape(num_tgt, self.num_frames, 4).unsqueeze(0)
            tgt_valid_split = tgt_valid.reshape(num_tgt, self.num_frames)
            frame_index = torch.arange(start=0, end=self.num_frames, device=tgt_valid.device).repeat(num_tgt).long()

            class_cost = -1 * out_prob_split[:, frame_index, tgt_ids].view(self.num_out, num_tgt, self.num_frames).mean(dim=-1)
            bbox_cost = (out_bbox_split - tgt_bbox_split).abs().mean((-1, -2))
            iou_cost = -1 * multi_iou(box_cxcywh_to_xyxy(out_bbox_split), box_cxcywh_to_xyxy(tgt_bbox_split)).mean(-1)
            # TODO: only deal with box and mask with empty target
            cost = self.cost_class * class_cost + self.cost_bbox * bbox_cost + self.cost_giou * iou_cost
            out_i, tgt_i = linear_sum_assignment(cost.cpu())
            index_i, index_j = [], []
            for j in range(len(out_i)):
                #Original VisTR implementation seems incorrect: valid doesn't pick real index of that match which is tgt_i[j], not j.
                tgt_valid_ind_j = tgt_valid_split[j].nonzero().flatten()
                # Correct implementation
                # tgt_valid_ind_j = tgt_valid_split[tgt_i[j]].nonzero().flatten()
                index_i.append(tgt_valid_ind_j * self.num_out + out_i[j])
                index_j.append(tgt_valid_ind_j + tgt_i[j] * self.num_frames)

            if index_i == [] or index_j == []:
                indices.append((torch.tensor([]).long().to(out_prob.device), torch.tensor([]).long().to(out_prob.device)))

            else:
                index_i = torch.cat(index_i).long()
                index_j = torch.cat(index_j).long()
                indices.append((index_i, index_j))

            if return_embd_idx:
                all_idxs = []
                for j in range(len(out_i)):
                    all_idxs.append(torch.arange(self.num_frames) * self.num_out + out_i[j])
                    valid_ordered_ind.append(tgt_valid_split[tgt_i[j]].bool())
                valid_ordered_ind = torch.cat(valid_ordered_ind)
                return indices, all_idxs, valid_ordered_ind
            else:
                return indices




class HungarianInferenceMatcher:

    def __init__(self, t_window: int = 2, cost_class: float = 2, cost_mask_iou: float = 6, score_cost: float = 2, cost_center_distance = 0,
                 use_frame_average_iou: bool = False, use_binary_mask_iou: bool = False, use_center_distance = False,):
        self.t_window = t_window
        self.class_cost = cost_class
        self.mask_iou_cost = cost_mask_iou
        self.score_cost = score_cost
        self.cost_center_distance = cost_center_distance
        self.use_frame_average_iou = use_frame_average_iou
        self.use_binary_mask_iou = use_binary_mask_iou
        self.use_center_distance = use_center_distance



    def compute_class_cost(self, track1: List, track2: List):
        cost_classes = []
        for t in range(self.t_window):
            classes_clip_1 = [track.get_last_t_result(-self.t_window + t, "categories") for track in track1]
            classes_clip_2 = [track.get_first_t_result(t, "categories") for track in track2]

            class_matrix = np.zeros((len(classes_clip_1), len(classes_clip_2)), dtype=np.float32)
            for idx_i, class_1 in enumerate(classes_clip_1):
                for idx_j, class_2 in enumerate(classes_clip_2):
                    # Assigns cost 1 if class equals
                    class_matrix[idx_i, idx_j] = class_1 == class_2 * 1.0

            cost_classes.append(class_matrix)
        total_cost_classes = np.stack(cost_classes, axis=0).mean(axis=0)

        return total_cost_classes

    def compute_score_cost(self, track1: List, track2: List):
        cost_score =  []
        for t in range(self.t_window):
            scores_clip_1 = [track.get_last_t_result(-self.t_window + t, "scores") for track in track1]
            scores_clip_2 = [track.get_first_t_result(t, "scores") for track in track2]

            score_matrix = np.zeros((len(scores_clip_1), len(scores_clip_2)), dtype=np.float32)
            for idx_i, score_1 in enumerate(scores_clip_1):
                for idx_j, score_2 in enumerate(scores_clip_2):
                    # Assigns cost 1 if class equals
                    score_matrix[idx_i, idx_j] = abs(score_1 - score_2)

            cost_score.append(score_matrix)

        total_cost_scores = np.stack(cost_score, axis=0).mean(axis=0)
        return total_cost_scores

    def compute_center_distance_cost(self, track1: List, track2: List):
        cost_ct =  []
        for t in range(self.t_window):
            centers_clip_1 = [track.get_last_t_result(-self.t_window + t, "centroid_points") for track in track1]
            centers_clip_2 = [track.get_first_t_result(t, "centroid_points") for track in track2]

            distance_matrix = np.zeros((len(centers_clip_1), len(centers_clip_2)), dtype=np.float32)
            for idx_i, center_1 in enumerate(centers_clip_1):
                for idx_j, center_2 in enumerate(centers_clip_2):
                    distance_matrix[idx_i, idx_j] = np.abs(np.array(center_1) - np.array(center_2)).sum()

            cost_ct.append(distance_matrix)

        total_cost_distances = np.stack(cost_ct, axis=0).mean(axis=0)
        return total_cost_distances

    def compute_frame_average_iou_cost(self, track1, track2):
        cost_iou = []
        for t in range(self.t_window):
            masks_clip_1 = [track.get_last_t_result(-self.t_window + t, "masks") for track in track1]
            masks_clip_2 = [track.get_first_t_result(t, "masks") for track in track2]
            if self.use_binary_mask_iou:
                iou_matrix = compute_iou_matrix(masks_clip_1, masks_clip_2, is_encoded=True)

            else:
                iou_matrix = np.zeros([len(track1), len(track2)])
                for i, j in np.ndindex(iou_matrix.shape):
                    iou_matrix[i, j] = self.soft_iou(masks_clip_1[i], masks_clip_2[j])
            cost_iou.append(iou_matrix)

        total_cost_iou = np.stack(cost_iou, axis=0).mean(axis=0)

        return total_cost_iou


    # Note that for soft_iou we need pixel probabilities, not binary masks
    @staticmethod
    def soft_iou(mask_logits1, mask_logits2):
        i, u = .0, .0
        if isinstance(mask_logits1, list):
            mask_logits1 = torch.stack(mask_logits1)
            mask_logits2 = torch.stack(mask_logits2)

        i += (mask_logits1 * mask_logits2).sum()
        u += (mask_logits1 + mask_logits2 - mask_logits1 * mask_logits2).sum().clamp(1e-6)
        iou = i / u if u > .0 else .0
        iou = iou.item()
        return iou

    @staticmethod
    def iou(track1_masks, track2_masks):
        i, u = .0, .0
        for d, g in zip(track1_masks, track2_masks):
            if d and g:
                i += mask_util.area(mask_util.merge([d, g], True))
                u += mask_util.area(mask_util.merge([d, g], False))
            elif not d and g:
                u += mask_util.area(g)
            elif d and not g:
                u += mask_util.area(d)
        if not u >= .0:
            print("UNION EQUALS 0")

        iou = i / u if u > .0 else .0
        return iou

    def compute_volumetric_iou_cost(self, track1: List, track2: List):
        ious = np.zeros([len(track1), len(track2)])
        track1_masks = [track.get_last_results(self.t_window, "masks") for track in track1]
        track2_masks = [track.get_first_results(self.t_window, "masks") for track in track2]
        track1_mask_ids = [track.get_mask_id() for track in track1]
        track2_mask_ids = [track.get_mask_id() for track in track2]
        iou_func = self.iou if self.use_binary_mask_iou else self.soft_iou
        iou_values_dict = {}
        for i, j in np.ndindex(ious.shape):
            combination_hash = f"{track1_mask_ids[i]}_{track2_mask_ids[j]}"
            if combination_hash not in iou_values_dict:
                iou_value = iou_func(track1_masks[i], track2_masks[j])
                ious[i, j] = iou_value
                iou_values_dict[combination_hash] = iou_value
            else:
                ious[i, j] = iou_values_dict[combination_hash]

        return ious

    def __call__(self, track1: List, track2: List):
        if self.use_frame_average_iou:
            total_cost_iou = self.compute_frame_average_iou_cost(track1, track2)
        else:
            total_cost_iou = self.compute_volumetric_iou_cost(track1, track2)

        total_cost_classes = self.compute_class_cost(track1, track2)
        total_cost_scores = self.compute_score_cost(track1, track2)

        cost = -1 * total_cost_iou * self.mask_iou_cost + -1 * total_cost_classes * self.class_cost + total_cost_scores * self.score_cost
        if self.cost_center_distance:
            total_cost_ct = self.compute_center_distance_cost(track1, track2)
            cost += self.cost_center_distance * total_cost_ct

        track1_ids, track2_ids = linear_sum_assignment(cost)


        return track1_ids, track2_ids