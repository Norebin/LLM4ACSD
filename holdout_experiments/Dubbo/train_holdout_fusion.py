import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

sys.path.insert(0, str(Path(os.environ["CODE_ROOT"]) / "RQ2"))
import fusion_model as fm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative = [0] + list(np.cumsum(self.lengths))
        self.semantic_dim = datasets[0].semantic_dim

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        dataset_index = np.searchsorted(self.cumulative[1:], index, side="right")
        return self.datasets[dataset_index][index - self.cumulative[dataset_index]]


target = os.environ["TARGET"]
train_projects = os.environ["TRAIN_PROJECTS_CSV"].split(",")
if target in train_projects:
    raise RuntimeError(f"Target project unexpectedly appears in training: {target}")

epochs = int(os.environ["EPOCHS"])
batch_size = 8
learning_rate = 5e-4
model_tag = os.environ["MODEL_TAG"]
result_dir = Path(os.environ["RESULT_DIR"])
result_dir.mkdir(parents=True, exist_ok=True)

common = dict(
    model_name=model_tag,
    semantic_feature_dir=os.environ["FEATURE_DIR"],
    csv_dir=os.environ["CSV_DIR"],
    graph_root_dir=os.environ["GRAPH_DIR"],
    semantic_feature_type="middle_avg_pooling",
)

train_sets = {
    project: fm.FusionDataset(project_name=project, split="train", **common)
    for project in train_projects
}
test_set = fm.FusionDataset(project_name=target, split="test", **common)
train_set = CombinedDataset(list(train_sets.values()))

generator = torch.Generator().manual_seed(seed)
train_loader = PyGDataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=fm.custom_collate,
    generator=generator,
)
test_loader = PyGDataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=fm.custom_collate,
)

sample_graph, _ = next(iter(train_loader))
fm.FusionClassifier.graph_input_dim = sample_graph.x.shape[1]
model = fm.FusionClassifier(
    semantic_dim=train_set.semantic_dim,
    graph_dim=64,
    fusion_dim=256,
    fusion_type="progress",
    gnn_type="graphsage",
).to(fm.DEVICE)

labels = np.array([train_set[index][0].y.item() for index in range(len(train_set))])
class_counts = np.bincount(labels, minlength=2)
if np.any(class_counts == 0):
    raise RuntimeError(f"Training set is missing a class: {class_counts.tolist()}")
weights = torch.tensor(
    len(labels) / (2.0 * class_counts), dtype=torch.float32, device=fm.DEVICE
)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

history = []
for epoch in range(1, epochs + 1):
    metrics = fm.train_epoch(model, train_loader, criterion, optimizer, fm.DEVICE)
    metrics["epoch"] = epoch
    history.append(metrics)
    print(f"Epoch {epoch}/{epochs} train: {metrics}", flush=True)

test_metrics, predictions = fm.evaluate(model, test_loader, criterion, fm.DEVICE)
tn, fp, fn, tp = confusion_matrix(
    predictions["true_label"], predictions["predicted_label"], labels=[0, 1]
).ravel()

weight_path = result_dir / f"{model_tag}_graphsage_progress_final.pt"
torch.save(model.state_dict(), weight_path)
pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)
predictions.to_csv(result_dir / f"predictions_{target}.csv", index=False)
pd.DataFrame([{**test_metrics, "target_project": target}]).to_csv(
    result_dir / f"metrics_{target}.csv", index=False
)

manifest = {
    "target_project": target,
    "train_projects": train_projects,
    "target_in_downstream_training": target in train_projects,
    "train_samples_by_project_after_graph_intersection": {
        project: len(dataset) for project, dataset in train_sets.items()
    },
    "total_train_samples_after_graph_intersection": len(train_set),
    "target_test_samples_after_graph_intersection": len(test_set),
    "semantic_model_reused": os.environ["SEMANTIC_MODEL"],
    "semantic_feature_type": "middle_avg_pooling",
    "gnn_type": "graphsage",
    "fusion_type": "progress",
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "seed": seed,
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "test_metrics": {key: float(value) for key, value in test_metrics.items()},
    "final_weight": str(weight_path),
    "protocol_note": (
        "Dubbo contributes zero downstream training samples. The semantic model is "
        "the handed-over all-project Phi-4 LoRA model, so this is downstream holdout, "
        "not complete semantic-stage LOPO."
    ),
}
(result_dir / "experiment_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
