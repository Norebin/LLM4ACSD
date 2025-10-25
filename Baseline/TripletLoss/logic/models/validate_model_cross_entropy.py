import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import copy
import tqdm
import math
from code_utils import peek
from logic import Label, SimpleClassifier, MultiLabelClassifier


# noinspection PyTypeChecker
class ValidationModelCrossEntropy:
    def __init__(self, label, dataset: data.Dataset, params, val_ratio=0, device=None):

        assert len(dataset) == len(label.labels)

        # labels
        assert isinstance(label, Label)
        self.label: Label = label

        assert params is not None
        self.params = params

        self.writer = None if "writer" not in self.params.keys() else self.params["writer"]

        # data
        self.batch_size = self.params["train_batch_size"]

        self.generator = torch.Generator().manual_seed(self.params["seed"]) # generator with seed

        dataset_and_labels = [(dataset[i], label.labels[i]) for i in range(len(dataset))]
        self.sampler = torch.utils.data.RandomSampler(dataset_and_labels, generator=self.generator)
        self.dataset = dataset
        self.loader = data.DataLoader(dataset_and_labels, batch_size=self.batch_size, sampler=self.sampler)

        # device
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # model
        self.classifier = None
        self.best_model = None

        self.val_ratio = val_ratio
        self.validate = (val_ratio != 0)

        self.history = None

    def __initialize__(self, multi_label=True):
        shape = peek(self.loader)[0].shape

        input_size_ = np.prod(shape[1:])  # Multiply shape values after batch size to get input size
        output_size_ = self.label.label_count  # Number of classes

        if not multi_label:
            self.classifier = SimpleClassifier(input_size_, output_size_)
        else:
            self.classifier = MultiLabelClassifier(input_size_, output_size_)

        self.classifier.to(self.device)
        self.classifier.train()

        self.weights = None if "weights" not in self.params.keys() else torch.tensor(self.params["weights"]).to(self.device)
        assert self.weights is None or len(self.weights) == output_size_
        # criterion = nn.CrossEntropyLoss(weight=self.weights)
        criterion = nn.BCELoss(weight=self.weights)
        if self.params["optimizer"] == "SGD":
            optimizer = optim.SGD(self.classifier.parameters(), lr=(self.params["lr"]))
        elif self.params["optimizer"] == "Adam":
            optimizer = optim.Adam(self.classifier.parameters(), lr=(self.params["lr"]), eps=1e-7)
        else:
            raise Exception("Optimizer given is wrong")

        return criterion, optimizer

    def train(self):
        criterion, optimizer = self.__initialize__()
        min_loss = 100
        num_epochs = self.params["num_epochs"]
        prev_loss = 100
        default_patience = 40 if "patience" not in self.params.keys() else self.params["patience"]
        tolerance = 1e-3 if "tolerance" not in self.params.keys() else self.params["tolerance"]
        patience = default_patience

        self.history = {
            "losses": [],
            "accuracy": [],
            "val_losses": [],
            "val_accuracy": [],
            "epoch_time": [],
            "general": {
                "classifier": str(self.classifier),
                "device": str(self.device),
                "dataset": str(self.dataset.__class__),
                "loader": str(self.loader.__class__),
                "tolerance": tolerance,
                "weights": str(None if self.weights is None else self.weights.detach().cpu().numpy().tolist()),
                "patience": default_patience,
                "max_epochs": num_epochs,
                "optimizer": str(optimizer),
                "criterion": str(criterion),
                "batch_size": self.batch_size,
                "class_count": self.label.labels.shape[1],
                "sample_size": len(self.dataset),
            }
        }

        self.best_model = copy.deepcopy(self.classifier)

        for epoch in range(num_epochs):
            total_loss = 0
            accuracy = 0

            val_count = 0
            count = 0

            if self.validate:
                val_accuracy = 0
                val_loss = 0

            optimizer.zero_grad()

            data_iter = tqdm.tqdm(enumerate(self.loader),
                                  desc="EP_%s:%d" % ("test", epoch),
                                  total=len(self.loader),
                                  bar_format="{l_bar}{r_bar}")

            for i, (data_, labels_) in data_iter:
                self.classifier.train()

                if self.validate:
                    split_pos = math.floor(len(data_) * self.val_ratio)
                    val_data, code_data = np.split(data_, [split_pos])
                    val_labels, code_labels = np.split(labels_, [split_pos])
                    val_target = val_labels.to(self.device).to(torch.float32)

                    val_count += len(val_data)

                else:
                    code_data = data_
                    code_labels = labels_

                count += len(code_data)

                code_target = code_labels.to(self.device).to(torch.float32)

                # TRAIN
                code_data_1d = torch.flatten(code_data, start_dim=1) if code_data.ndim > 2 else code_data
                outputs = self.classifier(code_data_1d)
                loss = criterion(outputs.squeeze(1), code_target[:, 1].float())

                loss.backward()
                total_loss += loss.item()

                # accuracy += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(code_target, dim=1)).item()
                accuracy += torch.sum((outputs > 0.5).long().squeeze(1) == torch.argmax(code_target, dim=1)).item()
                optimizer.step()

                # VALIDATE
                if self.validate:
                    self.classifier.eval()
                    with torch.no_grad():
                        val_data_1d = torch.flatten(val_data, start_dim=1) if val_data.ndim > 2 else val_data
                        outputs = self.classifier(val_data_1d)
                        # loss = criterion(outputs, val_target)
                        loss = criterion(outputs.squeeze(1), val_target[:, 1].float())
                        val_loss += loss.item() if not torch.isnan(loss) else 0
                        val_accuracy += torch.sum((outputs > 0.5).long().squeeze(1) == torch.argmax(val_target, dim=1)).item()
                        # val_accuracy += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(val_target, dim=1)).item()

            total_loss = total_loss / len(self.loader)
            accuracy = accuracy / count
            if self.writer:
                self.writer.add_scalar("Loss", total_loss, epoch)
                self.writer.add_scalar("Accuracy", accuracy, epoch)

            self.history["losses"].append(total_loss)
            self.history["accuracy"].append(accuracy)
            self.history["epoch_time"].append(data_iter.format_dict["elapsed"])

            if self.validate:
                val_loss = val_loss / len(self.loader)
                val_accuracy = val_accuracy / val_count
                if self.writer:
                    self.writer.add_scalar("Validation Loss", val_loss, epoch)
                    self.writer.add_scalar("Validation Accuracy", val_accuracy, epoch)

                self.history["val_losses"].append(val_loss)
                self.history["val_accuracy"].append(val_accuracy)

            if (epoch + 1) % 10 == 0:
                if self.validate:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy: .4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Patience: {patience}')
                else:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy: .4f}, Patience: {patience}')

            if self.validate and min_loss > (loss_inner := val_loss if self.validate else total_loss):
                min_loss = loss_inner
                self.best_model = copy.deepcopy(self.classifier)

            if total_loss > prev_loss - tolerance:
                if patience == 0:
                    print("Training finished early!")
                    self.history["early_finish"] = "True"
                    return self.best_model
                else:
                    patience -= 1
            else:
                prev_loss = total_loss
                patience = default_patience

        print("Training finished!")
        if self.writer:
            self.writer.flush()
        self.history["early_finish"] = "False"
        return self.best_model
