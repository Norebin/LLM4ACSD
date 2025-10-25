import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import logic
import logic.datasets
import logic.models
import code_utils
from pathlib import Path
import json


class Runner:
    def __init__(self, label_path, file_path, val_ratio, params, force=False, device=None, title=None, output_folder=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label_path = label_path
        self.file_path = file_path

        self.title = title if title is not None else Path(file_path).stem
        self.output_folder = output_folder

        self.val_ratio = val_ratio
        self.validate = (val_ratio != 0)

        assert params is not None
        self.params = params

        label_extension = os.path.splitext(label_path)[1]
        try:
            if label_extension == '.csv' or label_extension == '.json':
                self.labels = logic.Label(label_path)
            else:
                raise TypeError('Illegal label path')
        except KeyError as e:
            print("Smells should be under the key 'smellKey'")
            raise e

        file_extension = os.path.splitext(file_path)[1]
        if file_extension != '.npy' and not force:
            raise TypeError('Extension of data file should be .npy and the file should be embeddings saved using the '
                            'function \'numpy.save\'. If the file is a numpy array, change extension or set argument '
                            'force=True.')
        else:
            self.dataset = logic.datasets.LazyDataset(file_path, device)

    r'''Args: smell_range (int or (int, int)) : which smells to test based on value counts. Single parameter means 
    test only the x most frequent smells, tuple parameter means test the smells between x most frequent and y.'''
    def __pre_run(self, smells, shuffle):
        if isinstance(smells, tuple):
            assert len(smells) == 2
            smell_range = smells
        else:
            smell_range = (0, smells)

        label_series = self.labels.label_series
        most_freq = label_series.value_counts().index[smell_range[0]:smell_range[1]]
        top_indices = np.asarray(label_series[label_series.apply(lambda x: x in list(most_freq))].index)
        
        if shuffle:
            print(f"Current seed {np.random.get_state()[1][0]}")
            np.random.shuffle(top_indices)

        self.smell_range = smell_range
        self.smell_names = label_series.value_counts().index[smell_range[0]:smell_range[1]].tolist()

        return top_indices

    def __save(self, general_history, indices, history_list, predictions_list):
        final_history = {
            "smell_range": str(self.smell_range),
            "smell_names": str(self.smell_names),
            "label_path": str(os.path.abspath(self.label_path)),

            "file_path": str(os.path.abspath(self.file_path)),
            "parameters": {
                "val_ratio": str(self.val_ratio),
                "train_batch_size": str(self.params["train_batch_size"]),
                "lr": str(self.params["lr"])
            },

        }

        final_history.update(general_history)

        final_history["indices"] = indices
        final_history["folds"] = history_list

        now = code_utils.utils.get_now()

        if self.output_folder is not None:
            if not os.path.exists(self.output_folder):
                os.mkdir(self.output_folder)
            folder = os.path.join(self.output_folder, now) + "/"
        elif os.path.exists("results") and os.path.isdir("results"):
            folder = os.path.join("results", now) + "/"
        elif os.path.exists("../results") and os.path.isdir("../results"):
            folder = os.path.join("../results", now) + "/"
        else:
            folder = now + "/"

        folder = folder[:-1] + "_" + self.title + folder[-1]

        if os.path.exists(folder):
            if folder.rfind("/") + 1 == len(folder):  # if last char is /
                folder = folder[:-1]
            folder_iter = 2
            new_folder = folder + "_" + str(folder_iter)
            while os.path.exists(new_folder):
                folder_iter += 1
                new_folder = folder + "_" + str(folder_iter)
            folder = new_folder + "/"  # restore /

        os.mkdir(folder)

        save_history_at = folder + "metadata.json"
        save_at = folder + "predictions.npy"

        try:
            # np.save(open(save_at, "wb"), np.vstack(predictions_list))
            json.dump(final_history, open(save_history_at, "w"), indent=4)
        except Exception as e:
            print(e)
            print("Emergency dumping results and history...")
            print(final_history)
            np.save(open("emergency_dump.npy", "wb"), np.vstack(predictions_list))
            json.dump(final_history, open("emergency_history_dump.json", "w"), indent=4)
            raise e

        print("Prediction results saved at", save_at)
        print("Prediction metadata saved at", save_history_at)
        return folder

    def run(self, smells: int | tuple[int, int], shuffle=False, fold_size=5, test_batch_size=32):
        top_indices = self.__pre_run(smells, shuffle)

        folds = np.array_split(top_indices, fold_size)
        folds_as_array = np.asarray(folds, dtype=np.ndarray)
        predictions_list = []
        history_list = []
        general_history = None
        indices = []

        for i, _ in enumerate(folds):
            print(("-" * 20), "Start of fold", f"{i + 1}/{len(folds)}", ("-" * 20))

            folds_as_array = np.ma.array(folds_as_array, mask=False)
            folds_as_array.mask[i] = True

            train = folds_as_array.compressed() if len(top_indices) % fold_size == 0 else np.concatenate(folds_as_array.compressed())
            test = folds[i]
            indices.extend(test.tolist())

            data_train = logic.datasets.LazyDataset(self.file_path, self.device, indices=train)
            data_test = logic.datasets.LazyDataset(self.file_path, self.device, indices=test)

            train_target = logic.Label(self.labels.label_series.iloc[train])
            test_target = logic.Label(self.labels.label_series.iloc[test])

            # if "writer" not in self.params.keys():
            #     self.params["writer"] = SummaryWriter()

            model_ = logic.models.ValidationModelCrossEntropy(train_target, data_train, self.params,
                                                  self.val_ratio, self.device)

            model_.train()

            predicts, test_accuracy, precision, recall, f1, auc, mcc= code_utils.utils.predict(model_.best_model, DataLoader(data_test, batch_size=test_batch_size),
                                           test_target.labels, False)

            penultimate_history = {
                "fold": i,
                "test_accuracy": test_accuracy,
                'test_pre': precision,
                "test_r": recall,
                "test_f1": f1,
                "auc": auc,
                "mcc": mcc
            }

            general_history = model_.history["general"]

            model_.history.pop("general")

            penultimate_history.update(model_.history)

            # torch.save(model_.classifier, open("/home/eislamoglu/hmmmm.pth", "wb"))

            history_list.append(penultimate_history)

            folds_as_array.mask[i] = False
            predictions_list.append(predicts)

            if "writer" in self.params.keys():
                self.params["writer"].close()

        self.__save(general_history, indices, history_list, predictions_list)

    def run_single_fold(self, smells: int | tuple[int, int], shuffle=False, test_batch_size=32, ratio=0.25):
        top_indices = self.__pre_run(smells, shuffle)
        predictions_list = []
        history_list = []
        indices = []

        if 0 < ratio < 1:
            separator = np.floor(len(top_indices) * ratio).astype(int)
        elif ratio > 1:
            separator = ratio
        else:
            raise Exception("Illegal Ratio")

        train = np.arange(0, separator)
        test = np.arange(separator, len(top_indices))
        indices.extend(test.tolist())

        data_train = logic.datasets.LazyDataset(self.file_path, self.device, indices=train)
        data_test = logic.datasets.LazyDataset(self.file_path, self.device, indices=test)

        train_target = logic.Label(self.labels.label_series.iloc[train])
        test_target = logic.Label(self.labels.label_series.iloc[test])

        model_ = logic.models.ValidationModelCrossEntropy(train_target, data_train, self.params,
                                                              self.val_ratio, self.device)

        model_.train()

        predicts, test_accuracy = code_utils.utils.predict(model_.best_model,
                                                      DataLoader(data_test, batch_size=test_batch_size),
                                                      test_target.labels, False)

        penultimate_history = {
            "fold": 0,
            "test_accuracy": test_accuracy
        }

        general_history = model_.history["general"]

        model_.history.pop("general")

        penultimate_history.update(model_.history)

        history_list.append(penultimate_history)

        predictions_list.append(predicts)

        if "writer" in self.params.keys():
            self.params["writer"].close()

        self.__save(general_history, indices, history_list, predictions_list)
