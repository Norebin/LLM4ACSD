import itertools
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import copy
from code_utils.utils import save_annotation
import contextlib

param_keys = ["optimizer", "train_batch_size", "lr"]


def call_run(parameters):
    print("Starting run with parameters:", parameters)
    os.system(
        f'python parallel.py {parameters["embeds_path"]} {parameters["optimizer"]} '
        f'{parameters["train_batch_size"]} {parameters["lr"]} {parameters["label_path"]}')


def multi_embeds_parallel():
    embeds = [
         "/home/eislamoglu/PycharmProjects/siamese-triplet/results/triplet_2024_11_14__15_21_27_7500_smells_codebert_pooler_output/test_embeds_triplet_output.npy",
         "/home/eislamoglu/PycharmProjects/siamese-triplet/results/triplet_2024_11_14__15_21_32_7500_smells_codebert_pooler_output/test_embeds_triplet_output.npy",
         "/home/eislamoglu/PycharmProjects/siamese-triplet/results/triplet_2024_11_14__15_45_37_7500_smells_codebert_pooler_output/test_embeds_triplet_output.npy"
    ]

    parameters = {
        "optimizer": "Adam",
        "train_batch_size": 256,
        "lr": -4,
    }
    with ProcessPoolExecutor() as executor:
        for i, p in enumerate(embeds):
            current_params = copy.copy(parameters)
            current_params["embeds_path"] = p
            current_params["label_path"] = "/home/user/PycharmProjects/Model_Scratch/data/raw/7500_smells_test.json"

            executor.submit(call_run, current_params)


def hyper_parallel_batched(embeds_path, label_path, parameters=None):
    if parameters is None:
        parameters = {
            "optimizer": ["SGD", "Adam"],
            "train_batch_size": [32, 64, 128, 256, 512],
            "lr": np.arange(-6, -3, 1)
        }
    else:
        assert all(i in param_keys for i in parameters.keys())

    s = [np.arange(len(i)) for i in parameters.values()]
    perm = list(itertools.product(*s))

    perms_skip = [
        # ["SGD", 128, -6.0],
        # ["SGD", 64, -4.0],
        # ["SGD", 64, -6.0],
        # ["SGD", 64, -5.0],
        # ["SGD", 32, -5.0],
        # ["SGD", 32, -6.0],
        # ["SGD", 32, -4.0],
    ]

    # for i in range(0, len(perm), 7):
    #     curr_perms = perm[i * 7: (i + 1) * 7]

    futures = []
    with ProcessPoolExecutor(max_workers=7) as executor:
        for _, p in enumerate(perm):
            current_params = {
                "embeds_path": embeds_path,
                "label_path": label_path
            }
            for index, key in enumerate(parameters.keys()):
                current_params[key] = parameters[key][p[index]]

            skip = False
            for i in perms_skip:
                if current_params["optimizer"] == i[0] and current_params["train_batch_size"] == i[1] and current_params["lr"] == i[2]:
                    skip = True
                    break
            if skip:
                continue

            futures.append(executor.submit(call_run, current_params))
    wait(futures)
    save_annotation("hyperparam_run_" + os.path.basename(embeds_path), "Finished parallel hyperparam run for " + embeds_path)


def hyper_parallel(embeds_path, label_path, parameters=None):
    if parameters is None:
        parameters = {
            "optimizer": ["SGD", "Adam"],
            "train_batch_size": [32, 64, 128, 256, 512],
            "lr": np.arange(-6, -3, 1)
        }
    else:
        assert all(i in param_keys for i in parameters.keys())

    s = [np.arange(len(i)) for i in parameters.values()]
    perm = list(itertools.product(*s))

    futures = []
    with ProcessPoolExecutor() as executor:
        for _, p in enumerate(perm):
            current_params = {
                "embeds_path": embeds_path,
                "label_path": label_path
            }
            for index, key in enumerate(parameters.keys()):
                current_params[key] = parameters[key][p[index]]

            futures.append(executor.submit(call_run, current_params))

    wait(futures)
    save_annotation("hyperparam_run_" + os.path.basename(embeds_path), "Finished parallel hyperparam run for " + embeds_path)


if __name__ == "__main__":
    # embeds_paths = """data/1500_smells_bert_nli_mean_token_pooler_output.npy
    # data/1500_smells_codebert_pooler_output.npy
    # data/1500_smells_graphcodebert_pooler_output.npy""".split("\n")
    # graphcodebert_embeds = "data/1500_smells_graphcodebert_hidden_state.npy"
    #
    # label_path_ = "data/raw/7500_smells_test.json"
    #
    # # for i in embeds_paths:
    # #     print(i)
    # #     save_annotation("hyperparam_run_" + os.path.basename(i), "Starting 16 batch size only parallel hyperparam run for " + i)
    # #     parameters_ = {
    # #         "optimizer": ["SGD", "Adam"],
    # #         "train_batch_size": [16],
    # #         "lr": np.arange(-6, -3, 1)
    # #     }
    # #     hyper_parallel(i, label_path_, parameters_)
    #
    # save_annotation("hyperparam_run_" + os.path.basename(graphcodebert_embeds), "Starting parallel hyperparam run for " + graphcodebert_embeds)
    #
    # parameters_ = {
    #     "optimizer": ["SGD", "Adam"],
    #     "train_batch_size": [16],
    #     "lr": np.arange(-6, -3, 1)
    # }
    # hyper_parallel_batched(graphcodebert_embeds, label_path_, parameters_)
    multi_embeds_parallel()
