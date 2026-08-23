import json
import os
import shutil
from pathlib import Path

import pandas as pd
import torch

target = os.environ["TARGET"]
train_projects = os.environ["TRAIN_PROJECTS_CSV"].split(",")
source_csv_dir = Path(os.environ["SOURCE_CSV_DIR"])
source_feature_dir = Path(os.environ["SOURCE_FEATURE_DIR"])
csv_dir = Path(os.environ["CSV_DIR"])
feature_dir = Path(os.environ["FEATURE_DIR"])
run_root = Path(os.environ["RUN_ROOT"])
model_tag = os.environ["MODEL_TAG"]
feature_type = "middle_avg_pooling"


def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def locate_feature(project, split):
    preferred = source_feature_dir / f"lora_phi4_code_semantic_{project}_{split}.pt"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        path
        for path in source_feature_dir.glob(f"*{project}_{split}.pt")
        if "lora_phi4" in path.name and "dora" not in path.name
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one lora_phi4 feature for {project}/{split}, found "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def validate_payload(path):
    payload = load_payload(path)
    features = payload.get("features", {}).get(feature_type)
    ids = [str(value) for value in payload.get("metadata", {}).get("unique_ids", [])]
    if features is None or len(ids) != len(features):
        raise RuntimeError(f"Invalid feature payload: {path}")
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate semantic IDs in: {path}")
    return ids


source_files = {}
counts = {}

# Keep the handed-over CSV labels/splits. Only the train vectors of the other
# nine projects are exposed to the downstream trainer.
for project in train_projects:
    csv_source = source_csv_dir / f"{project}_Split.csv"
    shutil.copy2(csv_source, csv_dir / csv_source.name)
    feature_source = locate_feature(project, "train")
    ids = validate_payload(feature_source)
    feature_output = feature_dir / f"{model_tag}_code_semantic_{project}_train.pt"
    feature_output.symlink_to(feature_source.resolve())
    source_files[project] = {"train": str(feature_source)}
    counts[project] = {"train_feature_rows": len(ids)}

# Jedis contributes only its handed-over test partition. No Jedis train
# feature is linked into this run, so FusionDataset cannot load it for training.
target_csv = source_csv_dir / f"{target}_Split.csv"
shutil.copy2(target_csv, csv_dir / target_csv.name)
target_feature = locate_feature(target, "test")
target_ids = validate_payload(target_feature)
target_output = feature_dir / f"{model_tag}_code_semantic_{target}_test.pt"
target_output.symlink_to(target_feature.resolve())
source_files[target] = {"test": str(target_feature)}
counts[target] = {"test_feature_rows": len(target_ids)}

target_frame = pd.read_csv(target_csv)
if "unique_id" not in target_frame.columns:
    raise RuntimeError(f"Missing unique_id in {target_csv}")
target_csv_ids = set(target_frame["unique_id"].astype(str))
missing_csv_ids = sorted(set(target_ids).difference(target_csv_ids))
if missing_csv_ids:
    raise RuntimeError(
        f"Jedis test features contain {len(missing_csv_ids)} IDs absent from CSV"
    )

manifest = {
    "experiment": "jedis_downstream_holdout_reusing_handed_over_phi4",
    "target_project": target,
    "downstream_training_projects": train_projects,
    "target_in_downstream_training_projects": target in train_projects,
    "semantic_model_reused": os.environ["SEMANTIC_MODEL"],
    "llm_fine_tuning_rerun": False,
    "semantic_feature_extraction_rerun": False,
    "other_project_partition": "handed-over train split: 70% of post-SFT 60% pool",
    "target_partition": "handed-over test split: 30% of post-SFT 60% pool",
    "target_downstream_train_rows": 0,
    "target_test_feature_rows_before_graph_intersection": len(target_ids),
    "source_feature_files": source_files,
    "feature_counts_before_graph_intersection": counts,
    "seed": 42,
    "protocol_note": (
        "Jedis is excluded from downstream fusion training. The handed-over Phi-4 "
        "LoRA model was fine-tuned on all projects, so this is not complete LOPO at "
        "the semantic-model stage."
    ),
}
(run_root / "data_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
