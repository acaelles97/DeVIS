import json
import random

def read_json(dataset_json_path: str):
    with open(dataset_json_path, 'r') as fh:
        dataset = json.load(fh)
    return dataset

def save_json(dataset_json_path, new_dataset):
    with open(dataset_json_path, 'w') as anno_file:
        json.dump(new_dataset, anno_file)


def filter_random_videos_val(json_data, num_out_videos):
    videos = json_data["videos"]
    selected_videos = list(random.sample(videos, k=num_out_videos))
    json_data["videos"] = selected_videos

def filter_random_videos_train(json_data, num_out_videos):
    videos = json_data["videos"]
    selected_videos = list(random.sample(videos, k=num_out_videos))

    selected_videos_ids = [video["id"] for video in selected_videos]
    new_annots = []
    for annot in json_data["annotations"]:
        if annot["video_id"] not in selected_videos_ids:
            continue
        new_annots.append(annot)

    json_data["videos"] = selected_videos
    json_data["annotations"] = new_annots

    new_ids = 0
    conversion_dict = {}
    for video in json_data["videos"]:
        old_video_id = video["id"]
        conversion_dict[old_video_id] = new_ids
        video["id"] = new_ids
        new_ids += 1

    for annot in json_data["annotations"]:
        annot["video_id"] = conversion_dict[annot["video_id"]]


if __name__ == "__main__":
    train_path = "/usr/stud/cad/p028/data/Youtube_VIS/train/train.json"
    val_path = "/usr/stud/cad/p028/data/Youtube_VIS/valid/valid.json"
    out_val_path = "/usr/stud/cad/p028/data/Youtube_VIS/valid/mini_valid.json"
    out_train_path = "/usr/stud/cad/p028/data/Youtube_VIS/train/mini_train.json"

    train_dataset  = read_json(train_path)
    val_dataset = read_json(val_path)

    filter_random_videos_train(train_dataset, 10)
    filter_random_videos_val(val_dataset, 12)

    # new_dataset = filter_random_videos(dataset, num_out_videos)
    save_json(out_train_path, train_dataset)
    save_json(out_val_path, val_dataset)