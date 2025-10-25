import torch
import os
import numpy as np
from code_utils import set_seed
import run

import sys

if __name__ == "__main__":

    arguments = sys.argv

    device = torch.device("cuda")

    parameters = {
        "optimizer": arguments[2],
        "train_batch_size": int(arguments[3]),
        "lr": np.power(10., int(arguments[4])),
    }

    set_seed()
    print(f"Current seed {np.random.get_state()[1][0]}")
    print(">> Starting hyperparameter tuning run")

    embeds_path = arguments[1]
    label_path = arguments[5]

    current_params = {
        "seed": 42,
        "num_epochs": 2000,
        "patience": 2000
    }

    for index, key in enumerate(parameters.keys()):
        current_params[key] = parameters[key]

    output_folder = [i for i in os.listdir("results/") if i.startswith("hyperparam_run")]
    output_folder = "results/hyperparam_run" + str(len(output_folder) + 1)
    print(">> At folder", output_folder)

    print(current_params)
    runner = run.Runner(label_path, embeds_path, 0.2, device=device,
                        params=current_params, output_folder=output_folder)
    runner.run((0, 6), shuffle=True)
