import json
from collections import Counter
import torch

def read_json(path):
    with open(path, 'r') as submission_file:
        saved_results = json.load(submission_file)

    return saved_results


def check(determinism1, determinism2):
    for track in determinism1:
        if track["video_id"] not in counter_results1:
            counter_results1[track["video_id"]] = 0
        counter_results1[track["video_id"]] += 1

    for track in determinism2:
        if track["video_id"] not in counter_results2:
            counter_results2[track["video_id"]] = 0
        counter_results2[track["video_id"]] += 1



    matched_idx1 = []
    matched_idx2 = []
    for idx, video1 in enumerate(determinism1):
        already_matched = False
        for idx2, video2 in enumerate(determinism2):
            if already_matched:
                break
            if video2["video_id"] == video1["video_id"]:
                all_same_segms = []
                for segm1, segm2 in zip(video1["segmentations"], video2["segmentations"]):
                    if segm1 is None or segm2 is None:
                        if segm1 == segm2:
                            all_same_segms.append(True)
                    elif segm1["counts"] == segm2["counts"]:
                        all_same_segms.append(True)
                    else:
                        all_same_segms.append(False)

                if all(all_same_segms):
                    matched_idx1.append(idx)
                    matched_idx2.append(idx2)
                    already_matched= True

    for idx, video1 in enumerate(determinism1):
        if idx not in matched_idx1:
            print(f"Missing {idx}")

    print("A")

if __name__ == "__main__":
    # train_clip_1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/out_training_1/eval_results/epoch_1/raw_clip/CLIP_videoId_24_iter_4_process_0.pth", map_location='cpu')
    # train_out_1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/out_training_1/eval_results/epoch_1/raw_out/OUT_videoId_24_iter_0_process_0.pth", map_location='cpu')
    #
    # val_clip_1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/val_output_1/val_epoch_1/raw_clip/CLIP_videoId_24_iter_4_process_0.pth", map_location='cpu')
    # val_out_1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/val_output_1/val_epoch_1/raw_out/OUT_videoId_24_iter_0_process_0.pth", map_location='cpu')
    #
    # # torch.all(train_clip_1 == val_clip_1)
    #
    # torch.all(train_out_1["pred_logits"] == val_out_1["pred_logits"])
    # torch.all(train_out_1["pred_boxes"] == val_out_1["pred_boxes"])

    # train_clip_1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/out_training_1/eval_results/epoch_1/raw_clip/CLIP_videoId_242_iter_1_process_1.pth", map_location='cpu')
    # train_out_1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/out_training_1/eval_results/epoch_1/raw_out/OUT_videoId_242_iter_1_process_1.pth", map_location='cpu')
    #
    # val_clip_1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/val_output_1/val_epoch_1/raw_clip/CLIP_videoId_242_iter_1_process_1.pth", map_location='cpu')
    # val_out_1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/val_output_1/val_epoch_1/raw_out/OUT_videoId_242_iter_1_process_1.pth", map_location='cpu')
    #
    # # torch.all(train_clip_1 == val_clip_1)
    #
    # torch.all(train_out_1["pred_logits"] == val_out_1["pred_logits"])
    # torch.all(train_out_1["pred_boxes"] == val_out_1["pred_boxes"])

    # checkpoint1 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/indexes_0.pth", map_location='cpu')
    # checkpoint2 = torch.load("/usr/stud/cad/results/trainings/DefVisTr/debug_2/indexes_1.pth", map_location='cpu')

    # for key, value in checkpoint1['model'].items():
    #     if key not in checkpoint2["model"]:
    #         print("A")
    #     if not  torch.all(value == checkpoint2["model"][key]):
    #         print(f"{key} different value")
    #     else:
    #         print(f"{key} same value")
    epoch3 = read_json("/usr/stud/cad/results/trainings/DefVisTr/vistr_vs_ours_experiments/VisTRComparison_baseline_rudimentary_with_same_mask_head_TEnc-2-2_TDecSmart/eval_results/epoch_3/results.json")
    epoch4 = read_json("/usr/stud/cad/results/trainings/DefVisTr/vistr_vs_ours_experiments/VisTRComparison_baseline_rudimentary_with_same_mask_head_TEnc-2-2_TDecSmart/eval_results/epoch_4/results.json")
    epoch4_other = read_json("/usr/stud/cad/results/inference/vistr_vs_ours_experiments/VisTRComparison_baseline_rudimentary_with_same_mask_head_TEnc-2-2/val_epoch_4/results.json")


    # val_results_2 = read_json("/usr/stud/cad/results/trainings/DefVisTr/debug_2/val_output_1/val_epoch_1_process_1.json")


    # all1 = read_json("/usr/stud/cad/results/trainings/DefVisTr/debug_out/val_out/val_epoch_1/results.json")
    #
    # all2 = read_json("/usr/stud/cad/results/trainings/DefVisTr/debug_out/val_out/val_epoch_1/results.json")

    classes_dict = Counter()
    counter_results1 = {}
    counter_results2 = {}

    check(train_results, train_results)
    check(val_results, val_results_node14)
    #
    #
    # counter_results1 = {}
    # counter_results2 = {}
    #
    # track_results1 = read_json(
    #     "/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism7/track_results_0.json")
    # track_results2 = read_json(
    #     "/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism6/track_results_0.json")
    #
    # for track in track_results1:
    #     if track["video_id"] not in counter_results1:
    #         counter_results1[track["video_id"]] = 0
    #     counter_results1[track["video_id"]] += 1
    #
    # for track in track_results2:
    #     if track["video_id"] not in counter_results2:
    #         counter_results2[track["video_id"]] = 0
    #     counter_results2[track["video_id"]] += 1
    #
    # print("A")
    #
    #
    # counter_results1 = {}
    # counter_results2 = {}
    #
    # track_results1 = read_json(
    #     "/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism7/track_results_1.json")
    # track_results2 = read_json(
    #     "/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism6/track_results_1.json")
    #
    # for track in track_results1:
    #     if track["video_id"] not in counter_results1:
    #         counter_results1[track["video_id"]] = 0
    #     counter_results1[track["video_id"]] += 1
    #
    # for track in track_results2:
    #     if track["video_id"] not in counter_results2:
    #         counter_results2[track["video_id"]] = 0
    #     counter_results2[track["video_id"]] += 1
    #
    # print("A")
    #
    # counter_results1 = {}
    # counter_results2 = {}
    #
    # track_results1 = read_json(
    #     "/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism7/track_results_2.json")
    # track_results2 = read_json(
    #     "/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism6/track_results_2.json")
    #
    # for track in track_results1:
    #     if track["video_id"] not in counter_results1:
    #         counter_results1[track["video_id"]] = 0
    #     counter_results1[track["video_id"]] += 1
    #
    # for track in track_results2:
    #     if track["video_id"] not in counter_results2:
    #         counter_results2[track["video_id"]] = 0
    #     counter_results2[track["video_id"]] += 1
    #
    # print("A")

    # 111
    # raw1 = torch.load("/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism7/raw_results_0.pt")
    # raw2 = torch.load("/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism6/raw_results_0.pt")
    # keys = ["pred_logits", "pred_boxes", "pred_masks"]
    # for res1, res2 in zip(raw1, raw2):
    #     if not list(res1.keys())[0] == list(res2.keys())[0]:
    #         print("VIDEO ID DIFFERENT")
    #         continue
    #     for out1, out2 in zip(list(res1.values())[0], list(res2.values())[0]):
    #         for key in keys:
    #             if not torch.all(out1[key] == out2[key]):
    #                 print("AAAAA")
    #
    # raw1 = torch.load("/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism7/raw_results_1.pt", map_location="cpu")
    # raw2 = torch.load("/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism6/raw_results_1.pt", map_location="cpu")
    # keys = ["pred_logits", "pred_boxes", "pred_masks"]
    # for res1, res2 in zip(raw1, raw2):
    #     if not list(res1.keys())[0] == list(res2.keys())[0]:
    #         print("VIDEO ID DIFFERENT")
    #         continue
    #     for out1, out2 in zip(list(res1.values())[0], list(res2.values())[0]):
    #         for key in keys:
    #             if not torch.all(out1[key] == out2[key]):
    #                 print("AAAAA")
    #
    # raw1 = torch.load("/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism7/raw_results_2.pt", map_location="cpu")
    # raw2 = torch.load("/usr/stud/cad/results/DefVisTr_Inference/DefVisTR_Reset_3GPU_3DSegmHead/determinism_inference/epoch_10_determinism6/raw_results_2.pt", map_location="cpu")
    # keys = ["pred_logits", "pred_boxes", "pred_masks"]
    # for res1, res2 in zip(raw1, raw2):
    #     if not list(res1.keys())[0] == list(res2.keys())[0]:
    #         print("VIDEO ID DIFFERENT")
    #         continue
    #     for out1, out2 in zip(list(res1.values())[0], list(res2.values())[0]):
    #         for key in keys:
    #             if not torch.all(out1[key] == out2[key]):
    #                 print("AAAAA")