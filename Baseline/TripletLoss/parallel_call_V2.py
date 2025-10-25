import itertools
import os
from asyncio import as_completed
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np

param_keys = ["optimizer", "train_batch_size", "lr"]


def call_run(call_str):
    print(call_str)
    os.system(call_str)


def call_parallel(embeds_paths, label_path, parameters=None, max_workers=None, non_perm_parameters=None, wait_end=True):
    if non_perm_parameters is None:
        non_perm_parameters = {}

    if parameters is None:
        parameters = {
            "optimizer": ["SGD", "Adam"],
            "train_batch_size": [32, 64, 128, 256, 512],
            "lr": np.arange(-6, -3, 1)
        }
    else:
        assert all(i in parameters.keys() for i in param_keys)

    if "output_folder" not in non_perm_parameters.keys():
        output_folder = [i for i in os.listdir("results/") if i.startswith("hyperparam_run")]
        output_folder = "results/hyperparam_run" + str(len(output_folder) + 1)
    else:
        output_folder = non_perm_parameters["output_folder"]
    print(">> At folder", output_folder)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for embeds_path in embeds_paths:
            s = [np.arange(len(i)) for i in parameters.values()]
            perm = list(itertools.product(*s))
            for _, p in enumerate(perm):
                current_params = {
                    "embeds_path": embeds_path,
                    "label_path": label_path,
                    "smell_range": ",".join(str(i) for i in (0, 7)),
                    "shuffle": 1,
                    "num_epochs": 2000,
                    "patience": 2000
                }
                current_params = current_params | non_perm_parameters
                for index, key in enumerate(parameters.keys()):
                    current_params[key] = parameters[key][p[index]]

                call_str = (f"python run_test.py -ep {embeds_path} -lp {label_path} -op {current_params['optimizer']} "
                            f"-lr {current_params['lr']} -bs {current_params['train_batch_size']} "
                            f"-sr {current_params['smell_range']} -sh {current_params['shuffle']} "
                            f"-e {current_params['num_epochs']} -pt {current_params['patience']} "
                            f"-o {output_folder}")
                futures.append(executor.submit(call_run, call_str))
        if wait_end:
            wait(futures)
        else:
            executor.shutdown(False, cancel_futures=False)


def call_siamese_triplets():
    embeds_folders = """/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__17_37_00_9000_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__17_39_24_9000_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__17_53_16_9000_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__18_44_45_9000_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__18_53_24_9000_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__19_01_49_9000_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__20_57_50_9000_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__21_33_32_9000_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__21_39_24_9000_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__21_54_33_9000_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__22_02_08_9000_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__22_04_03_9000_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__22_17_34_9000_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__22_28_38_9000_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_04_34_9000_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_08_23_9000_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_16_39_9000_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_17_56_9000_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_24_50_9000_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_25_27_9000_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_39_13_9000_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_45_56_9000_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_20__23_47_10_9000_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/27/triplet_2024_12_21__00_07_20_9000_smells_codebert_pooler_output""".split("\n")

    embed_type = "test"
    embed_name = embed_type + "_embeds_triplet_output.npy"
    label_path = f"/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_{embed_type}.json"

    embeds_paths = []
    for i in embeds_folders:
        embeds_paths.append(os.path.join(i, embed_name))

    parameters = {
        "optimizer": ["Adam"],
        "train_batch_size": [256],
        "lr": [-4],
    }

    non_perm_params = {
        "output_folder": "results/revision/v6_online"
    }

    call_parallel(embeds_paths, label_path, parameters=parameters, non_perm_parameters=non_perm_params, max_workers=24)


def call_offline():
    java_embeds = """/home/user/Desktop/Triplet-net-keras/Test/revision2/java_bert_nli_mean_token_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision2/java_codebert_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision2/java_graphcodebert_pooler_output.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_bert_nli_mean_token_pooler_output_java.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_codebert_pooler_output_java.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_graphcodebert_pooler_output_java.npy""".split("\n")

    php_embeds = """/home/user/Desktop/Triplet-net-keras/Test/revision2/php_bert_nli_mean_token_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision2/php_codebert_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision2/php_graphcodebert_pooler_output.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_bert_nli_mean_token_pooler_output_php.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_codebert_pooler_output_php.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_graphcodebert_pooler_output_php.npy""".split("\n")

    py_embeds = """/home/user/Desktop/Triplet-net-keras/Test/revision2/python_bert_nli_mean_token_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision2/python_codebert_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision2/python_graphcodebert_pooler_output.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_bert_nli_mean_token_pooler_output_python.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_codebert_pooler_output_python.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_graphcodebert_pooler_output_python.npy""".split("\n")

    gb_java = """/home/user/Desktop/Triplet-net-keras/Test/revision2/java_graphcodebert_hidden_state.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_graphcodebert_hidden_state_java.npy""".split("\n")

    gb_php = """/home/user/Desktop/Triplet-net-keras/Test/revision2/php_graphcodebert_hidden_state.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_graphcodebert_hidden_state_php.npy""".split("\n")

    gb_py = """/home/user/Desktop/Triplet-net-keras/Test/revision2/python_graphcodebert_hidden_state.npy
/home/user/PycharmProjects/Model_Scratch/data/revision_v2/9000_smells_graphcodebert_hidden_state_python.npy""".split("\n")

    parameters = {
        "optimizer": ["Adam"],
        "train_batch_size": [256],
        "lr": [-4],
    }

    non_perm_parameters = {
        "output_folder": "results/revision/v7_offline"
    }

    java_labels_path = "data/raw/9000_smells_test_java.json"
    call_parallel(gb_java, java_labels_path, parameters=parameters, max_workers=1, non_perm_parameters=non_perm_parameters, wait_end=False)

    php_labels_path = "data/raw/9000_smells_test_php.json"
    call_parallel(gb_php, php_labels_path, parameters=parameters, max_workers=1, non_perm_parameters=non_perm_parameters, wait_end=False)

    py_labels_path = "data/raw/9000_smells_test_python.json"
    call_parallel(gb_py, py_labels_path, parameters=parameters, max_workers=1, non_perm_parameters=non_perm_parameters, wait_end=False)


def call_combined():
    triplet_embeds = """/model/lxj/Baseline/TripletLoss/data/triple/actionableSmell_bert_nli_mean_token_pooler_output.npy
/model/lxj/Baseline/TripletLoss/data/triple/actionableSmell_codebert_pooler_output.npy
/model/lxj/Baseline/TripletLoss/data/triple/actionableSmell_graphcodebert_pooler_output.npy""".split("\n")

    original_embeds = """/home/user/Desktop/Triplet-net-keras/Test/revision2/all_langs_bert_nli_mean_token_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision2/all_langs_codebert_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision2/all_langs_graphcodebert_pooler_output.npy""".split("\n")

    gb_embeds = [
        "/home/user/Desktop/Triplet-net-keras/Test/revision2/all_langs_graphcodebert_hidden_state.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_hidden_state_test.npy"
    ]

    parameters = {
        "optimizer": ["Adam"],
        "train_batch_size": [256],
        "lr": [-4],
    }

    non_perm_parameters = {
        "output_folder": "results/revision/v7_offline"
    }

    label_path = "/model/lxj/Baseline/TripletLoss/data/labels.csv"
    # call_parallel(original_embeds, label_path, parameters, non_perm_parameters=non_perm_parameters, wait_end=False)
    call_parallel(triplet_embeds, label_path, parameters, non_perm_parameters=non_perm_parameters, wait_end=False)
    # call_parallel(gb_embeds, label_path, parameters, max_workers=1, non_perm_parameters=non_perm_parameters, wait_end=False)


def call_hyperparam():
    original_embeds = [
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_bert_nli_mean_token_pooler_output_test.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_codebert_pooler_output_test.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_pooler_output_test.npy"
    ]

    gb_embeds = ["/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_hidden_state_test.npy"]

    label_path = "data/raw/9000_smells_test.json"

    call_parallel(original_embeds, label_path, None, max_workers=90, wait_end=True)

    call_parallel(gb_embeds, label_path, None, max_workers=6)


if __name__ == '__main__':
    call_combined()
