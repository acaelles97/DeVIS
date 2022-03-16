import pickle
import os


def read_files(directory, files):

    info_dir = {}
    for file in files:
        with open(os.path.join(directory, file), 'rb') as f:
            mynewlist = pickle.load(f)

        min_len = min([len(list_time) for list_time in mynewlist])
        flattened_values = []
        for idx in range(min_len):
            # flattened_values.append(sum([list_time[idx] for list_time in mynewlist])/len(mynewlist))
            # flattened_values.append(mynewlist[0][idx])

            for list_time in mynewlist:
                flattened_values.append(list_time[idx])

        info_dir[file.split("_")[-2]] = flattened_values

    return info_dir




if __name__ == "__main__":
    large_embd_ovis = "/usr/prakt/p028/projects/results/VisTR_trainings/speed_experiments/30_1324_embd_ovis"
    ovis_original_embd_ = "/usr/prakt/p028/projects/results/VisTR_trainings/speed_experiments/original_embeddings_ovis"
    yvis_original_embd = "/usr/prakt/p028/projects/results/VisTR_trainings/speed_experiments/original_embeddings_yvis"
    single_gpu = "/usr/prakt/p028/projects/results/VisTR_trainings/speed_experiments/30_1324_embd_ovis_single_gpu"
    quadruple_gpu = "/usr/prakt/p028/projects/results/VisTR_trainings/speed_experiments/30_1324_embd_ovis_quadruple_gpu"
    files_to_read = ["all_inference_time.pkl", "all_optimization_time.pkl", "all_preprocessing_time.pkl"]

    large_embd_ovis_info = read_files(quadruple_gpu, files_to_read)

    # for i in large_embd_ovis_info["preprocessing"]:
    #     print(i)
    #
    # for i in large_embd_ovis_info["inference"]:
    #     print(i)
    # #
    for i in large_embd_ovis_info["optimization"]:
        print(i)

