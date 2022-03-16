import random
import json
import copy

def read_json(dataset_json_path: str):
    with open(dataset_json_path, 'r') as fh:
        dataset = json.load(fh)
    return dataset


def save_json(dataset, dataset_json_path: str):
    with open(dataset_json_path, 'w') as file:
        json.dump(dataset, file)



def split_dataset(dataset, train_split_perc):
    all_videos_ids = [video["id"] for video in dataset["videos"]]
    random.shuffle(all_videos_ids)

    idx = int(len(all_videos_ids) * train_split_perc)
    train_videos_ids, val_videos_ids = all_videos_ids[:idx], all_videos_ids[idx:]

    val_videos = [video for video in dataset["videos"] if video["id"] in val_videos_ids]
    train_videos = [video for video in dataset["videos"] if video["id"] in train_videos_ids]

    val_annots = [annot for annot in dataset["annotations"] if annot["video_id"] in val_videos_ids]
    train_annots = [annot for annot in dataset["annotations"] if annot["video_id"] in train_videos_ids]

    return train_annots, train_videos, val_annots, val_videos

if __name__ == "__main__":
    json_file = "/usr/stud/cad/p028/data/Youtube_VIS/train/train.json"
    json_file_out_train = "/usr/stud/cad/p028/data/Youtube_VIS/train/train_train_val_split.json"
    json_file_out_val = "/usr/stud/cad/p028/data/Youtube_VIS/train/valid_train_val_split.json"

    dataset = read_json(json_file)

    train_annots, train_videos, val_annots, val_videos = split_dataset(dataset, 0.85)

    dataset["annotations"] = train_annots
    dataset["videos"] = train_videos
    save_json(dataset, json_file_out_train)

    dataset["annotations"] = val_annots
    dataset["videos"] = val_videos
    save_json(dataset, json_file_out_val)






