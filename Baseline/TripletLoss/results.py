import pathlib
import zipfile
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd
import os

import sklearn.metrics
from matplotlib.ticker import NullFormatter, FixedLocator
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import logic
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib as pl
from enum import Enum

from triplet_net.tsne import TSNECreator
import code_utils.utils
import sklearn.metrics as metrics


class Metric(Enum):
    LOSS = "loss"
    ACCURACY = "accuracy"


def init_axis(nrows, ncols, figsize=(10, 5), height_ratios=None, pad=None, fontsize=15, xrotate=None):
    scale = 0.75
    figsize = tuple(i * scale for i in figsize)
    fontsize *= scale
    if pad is not None:
        pad *= scale
    fig_, ax_ = plt.subplots(nrows, ncols, figsize=figsize, height_ratios=height_ratios)
    ax_: list[plt.Axes] = ax_
    if pad is not None:
        fig_.tight_layout(pad=pad)
    for i in np.array(ax_).ravel():
        i.tick_params(axis='x', labelsize=fontsize, rotation=xrotate)
        i.tick_params(axis='y', labelsize=fontsize)
        i.set_xlabel(i.get_xlabel(), fontsize=fontsize)
        i.set_ylabel(i.get_ylabel(), fontsize=fontsize)
    return fig_, ax_


class Result:
    @staticmethod
    def get_latest(folder, at=-1):
        files = []
        mtimes = []

        for i in os.listdir(folder):
            current_file = pl.Path(os.path.join(os.path.abspath(folder), i))

            if i.startswith("2024_"):
                current_time = current_file.stat().st_mtime
                files.append(current_file)
                mtimes.append(current_time)

        return Result(np.array(files)[np.array(mtimes).argsort()][at])

    def __init__(self, result_folder_path):
        self.result_folder_path = result_folder_path

        self._predictions_path = os.path.join(self.result_folder_path, "predictions.npy")
        self._metadata_path = os.path.join(self.result_folder_path, "metadata.json")

        self.metadata = json.load(open(self._metadata_path, "r"))

        self.label_path = self.metadata["label_path"]

        self.labels = logic.Label(self.label_path)

        self.results_matrix = np.load(self._predictions_path)
        results_array = np.argmax(self.results_matrix, axis=1)

        tested_labels = logic.Label(self.labels.label_series.iloc[self.metadata["indices"]])
        labels_array = np.argmax(tested_labels.labels, axis=1)

        assert results_array.shape[0] == tested_labels.labels.shape[0]

        self.predicted = pd.Series(results_array, name="predicted")
        self.target = pd.Series(labels_array, name="target")

        self.result_df = pd.concat([self.predicted, self.target], axis=1)

        self.class_count = len(self.target.unique())
        self.foldc = len(self.metadata["folds"])

    def _accuracy_score(self):
        return accuracy_score(self.target, self.predicted)

    def _precision_score(self):
        return precision_score(self.target, self.predicted, average="macro")

    def _recall_score(self):
        return recall_score(self.target, self.predicted, average="macro")

    def _f1_score(self):
        return f1_score(self.target, self.predicted, average="macro")

    def _accuracy_score_per_class(self):
        accuracy = np.zeros(self.class_count, dtype=np.float32)
        unique_vals = self.target.unique()
        for i in unique_vals:
            accuracy[i] = (self.target[(cond:=self.target == i)] == self.predicted[cond]).sum() / sum(cond)
        return pd.Series(accuracy, name="Accuracy")

    def _precision_score_per_class(self):
        return pd.Series([precision_score(self.target, self.predicted, average="macro", labels=[i], zero_division=0)
                          for i in range(self.class_count)], name="Precision")

    def _recall_score_per_class(self):
        return pd.Series([recall_score(self.target, self.predicted, average="macro", labels=[i])
                          for i in range(self.class_count)], name="Recall")

    def _f1_score_per_class(self):
        return pd.Series([f1_score(self.target, self.predicted, average="macro", labels=[i])
                          for i in range(self.class_count)], name="F1")

    def get_confusion_matrix(self, rename_cols=False):
        crosstab = pd.crosstab(self.target, self.predicted, margins=True)
        if rename_cols:
            crosstab.columns = list(self.labels.le.inverse_transform([int(i) for i in crosstab.columns[:-1]])) + ["Target"]
            crosstab.index = list(self.labels.le.inverse_transform([int(i) for i in crosstab.index[:-1]])) + ["Predicted"]
        return crosstab

    def get_metrics_per_class_matrix(self, rename_cols=False):
        """metrics are accuracy, precision, recall, f1"""
        accuracy = self._accuracy_score_per_class()
        precision = self._precision_score_per_class()
        recall = self._recall_score_per_class()
        f1 = self._f1_score_per_class()

        df = pd.concat([accuracy, precision, recall, f1], axis=1).T
        if rename_cols:
            df = df.rename(columns={i:j for i, j in enumerate(self.labels.le.inverse_transform(df.columns))})
        return df

    def get_metrics(self):
        accuracy = self._accuracy_score()
        precision = self._precision_score()
        recall = self._recall_score()
        f1 = self._f1_score()
        series = pd.Series([accuracy, precision, recall, f1], index=["accuracy", "precision", "recall", "f1"])
        return series

    def plot_result_graph(self):
        df_melted = pd.concat([self.target.value_counts()._set_name("target"),
                               self.predicted.value_counts()._set_name("predicted")],
                              axis=1).reset_index().melt(id_vars="index")
        _, ax = init_axis(1, 1, figsize=(15, 7.5))
        sns.barplot(df_melted, x="index", y="value", hue="variable", ax=ax)
        plt.show()

    def total_accuracy(self):
        return self._accuracy_score()

    def reflect_for_fold(self, title: Metric, is_val):
        if title == Metric.LOSS:
            if is_val:
                return self.val_loss_per_epoch_for_fold
            else:
                return self.loss_per_epoch_for_fold
        elif title == Metric.ACCURACY:
            if is_val:
                return self.val_acc_per_epoch_for_fold
            else:
                return self.acc_per_epoch_for_fold
        else:
            raise ValueError("Illegal title")

    def loss_per_epoch_for_fold(self, fold):
        fold_data = self.metadata["folds"][fold]
        losses = fold_data["losses"]
        return losses

    def val_loss_per_epoch_for_fold(self, fold):
        fold_data = self.metadata["folds"][fold]
        losses = fold_data["val_losses"]
        return losses

    def acc_per_epoch_for_fold(self, fold):
        fold_data = self.metadata["folds"][fold]
        losses = fold_data["accuracy"]
        return losses

    def val_acc_per_epoch_for_fold(self, fold):
        fold_data = self.metadata["folds"][fold]
        losses = fold_data["val_accuracy"]
        return losses

    def _extract_fold(self, fold):
        fold_indices = np.cumsum(
            np.pad([len(i) for i in np.array_split(np.arange(len(self.result_df["target"])), self.foldc)], (1, 0),
                   "constant"))
        return self.result_df.iloc[fold_indices[fold]:fold_indices[fold + 1]]

    def get_class_ratios_for_fold(self, fold):
        fold_df = self._extract_fold(fold)
        fold_df = fold_df["target"].value_counts(normalize=True)
        return fold_df


def print_result(lh_result: Result, rh_result: Result, lh_title=None, rh_title=None):
    save_path = "results/print/" + code_utils.get_now() + ".txt"
    with open(save_path, "w") as f:
        with redirect_stdout(f):
            print("LEFTHAND" if lh_title is None else lh_title.upper())
            print("PATH", lh_result.result_folder_path)
            print("ACC", lh_result.total_accuracy())
            print("PRINT")
            print(lh_result.get_confusion_matrix())
            print()
            print("RIGHTHAND" if rh_title is None else rh_title.upper())
            print("PATH", rh_result.result_folder_path)
            print("ACC", rh_result.total_accuracy())
            print("PRINT")
            print(rh_result.get_confusion_matrix())

    return save_path


def plot_result(metric: Metric, do_val, lh_result, rh_result, lh_title=None, rh_title=None, plot_suptitle=None, ax=None, save=False):
    def forward(a):
        return np.power(np.abs(a), 1 / 3)

    def inverse(a):
        return np.power(a, 3)

    def style(ax_, epoch_count:int, title=None):

        ax_.set_xscale("function", functions=(forward, inverse))
        ax_.xaxis.set_minor_formatter(NullFormatter())

        ax_.set_xlim([0, epoch_count])
        xticks = np.array([epoch_count // 80, epoch_count // 20, epoch_count // 2, epoch_count])

        ax_.xaxis.set_major_locator(FixedLocator(xticks))
        ax_: plt.Axes = ax_

        if metric == Metric.LOSS:
            ax_.legend(fontsize=13, loc="upper right")
        elif metric == Metric.ACCURACY:
            ax_.legend(fontsize=13, loc="lower right")
        if title is not None:
            ax_.set_title(title, fontdict={"fontsize": 16})

    if ax is None:
        _, ax = init_axis(1, 1, (12, 8.5), fontsize=18)

    lh_title = lh_title if lh_title is not None else "lefthand embeddings"
    rh_title = rh_title if rh_title is not None else "righthand embeddings"

    assert metric == Metric.LOSS or metric == metric.ACCURACY

    # get minimum epoch count between all folds
    fold_count_min_lh = np.min([len(lh_result.reflect_for_fold(metric, False)(i)) for i in range(5)])
    fold_count_min_rh = np.min([len(rh_result.reflect_for_fold(metric, False)(i)) for i in range(5)])
    fold_count_min = min(fold_count_min_lh, fold_count_min_rh)

    lh_values = np.mean(
        np.array([lh_result.reflect_for_fold(metric, False)(i)[:fold_count_min] for i in range(5)]), axis=0)
    rh_values = np.mean(
        np.array([rh_result.reflect_for_fold(metric, False)(i)[:fold_count_min] for i in range(5)]), axis=0)

    # only plot the best value up to that point
    if metric == Metric.LOSS:
        lh_values = np.minimum.accumulate(lh_values)
        rh_values = np.minimum.accumulate(rh_values)
    else:
        lh_values = np.maximum.accumulate(lh_values)
        rh_values = np.maximum.accumulate(rh_values)

    plot_title = metric.value

    sns.lineplot(lh_values, label=lh_title + " train " + plot_title, linewidth=1, ax=ax)
    sns.lineplot(rh_values, label=rh_title + " train " + plot_title, linewidth=1, ax=ax)

    if do_val:
        lh_val_values = np.mean(
            np.array([lh_result.reflect_for_fold(metric, True)(i)[:fold_count_min] for i in range(5)]), axis=0)
        rh_val_values = np.mean(
            np.array([rh_result.reflect_for_fold(metric, True)(i)[:fold_count_min] for i in range(5)]),
            axis=0)
        sns.lineplot(lh_val_values, label=lh_title + " validation " + plot_title, ax=ax)
        sns.lineplot(rh_val_values, label=rh_title + " validation " + plot_title, ax=ax)

    if plot_suptitle is None:
        plot_suptitle = "Training " + plot_title.title() + " per Epoch"
    style(ax, fold_count_min, plot_suptitle)

    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)

    if save:
        now = code_utils.get_now()
        save_path = ("/home/eislamoglu/Pictures/accs/accuracy_graph_" + now
                     if metric == Metric.ACCURACY else "/home/eislamoglu/Pictures/losses/loss_graph_" + now) + ".png"
        plt.savefig(save_path, dpi=500)
    else:
        save_path = None

    return save_path, ax


def save_results(result_lh: Result, result_rh: Result, lh_title=None, rh_title=None, do_val=False, do_tsne=False):
    acc_graph = plot_result(Metric.ACCURACY, do_val, result_lh, result_rh, lh_title, rh_title)
    loss_graph = plot_result(Metric.LOSS, do_val, result_lh, result_rh, lh_title, rh_title)
    result_path = print_result(result_lh, result_rh, lh_title, rh_title)

    if do_tsne:
        lh_path = result_lh.metadata["file_path"]
        rh_path = result_rh.metadata["file_path"]
        lh_tsne, rh_tsne = TSNECreator.create_from_pair(lh_path, rh_path, result_lh.label_path)
        compress(acc_graph, loss_graph, result_path, lh_tsne, rh_tsne, zipfile_name=pathlib.Path(result_lh.metadata["file_path"]).stem)

    else:
        compress(acc_graph, loss_graph, result_path, zipfile_name=pathlib.Path(result_lh.metadata["file_path"]).stem)


def compress(*file_names, zipfile_name):
    print("File Paths:")
    print(file_names)

    # Select the compression mode ZIP_DEFLATED for compression
    # or zipfile.ZIP_STORED to just store the file
    compression = zipfile.ZIP_DEFLATED

    # create the zip file first parameter path/name, second mode
    now = code_utils.get_now()
    zf = zipfile.ZipFile("results/archives/" + now + "_" + zipfile_name + ".zip", mode="w")
    try:
        for file_name in file_names:
            # Add file to the zip file
            # first parameter file to zip, second filename in zip
            zf.write(file_name, os.path.basename(file_name), compress_type=compression)
        print("Zip file created at:", "results/archives/" + now + "_" + zipfile_name + ".zip")
    except FileNotFoundError as e:
        print("An error occurred")
        raise e
    finally:
        zf.close()
