# Modified for METAN portable release 2026-11-09.
"""Per-task pooling export for independent models; never invent shared feature maps."""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch


def save_independent_prediction_outputs(outputs, dataset, output_dir, stage, phenotype_names):
    folder = Path(output_dir)
    csv_path = folder / f"{stage}_predictions.csv"
    h5_path = folder / f"{stage}_task_pooling_weights.h5"
    if csv_path.exists() or h5_path.exists():
        raise FileExistsError("Independent export refuses to overwrite existing prediction/pooling files")
    preds = torch.cat([out["preds"].detach().cpu() for out in outputs]).float().numpy()
    labels = torch.cat([out["labels"].detach().cpu() for out in outputs]).float().numpy()
    if preds.shape != labels.shape or preds.shape != (len(dataset), len(phenotype_names)):
        raise ValueError("Independent export prediction/label/dataset shape mismatch")
    if not np.isfinite(preds).all() or not np.isfinite(labels).all():
        raise ValueError("Independent export contains non-finite prediction/label values")
    data = {}
    base = getattr(dataset, "dataset", dataset)
    ids = getattr(base, "sample_ids", None)
    if ids is not None:
        indices = getattr(dataset, "indices", range(len(dataset)))
        selected = [ids[int(i)] for i in indices]
        selected = [x.decode() if isinstance(x, bytes) else str(x) for x in selected]
        if len(set(selected)) != len(dataset):
            raise ValueError("Independent export requires unique sample IDs")
        data["sample_id"] = selected
    for i, trait in enumerate(phenotype_names):
        data[trait + "_pred"] = preds[:, i]
        data[trait + "_label"] = labels[:, i]
    for out in outputs:
        if out.get("phenotype_names") != list(phenotype_names):
            raise ValueError("Independent export trait order mismatch")
        if set(out.get("task_feature_exports", {})) != set(phenotype_names):
            raise ValueError("Independent export is missing per-task feature metadata")
    folder.mkdir(parents=True, exist_ok=True)
    # Pooling tensors retain their original dimensions. No concatenated task map is fabricated.
    with h5py.File(h5_path, "x") as hf:
        hf.attrs["task_sharing_mode"] = "independent"
        hf.attrs["layout"] = "task index / batch index / original pooling tensors"
        hf.attrs["prediction_scale"] = "model output and datamodule label scale; not automatically inverse-standardized"
        hf.attrs["attention_export"] = "full attention and feature tensors are available via predict_step, not duplicated in this pooling file"
        for i, trait in enumerate(phenotype_names):
            task = hf.create_group(f"task_{i}")
            task.attrs["phenotype_name"] = trait
            for batch_idx, out in enumerate(outputs):
                export = out["task_feature_exports"][trait]
                group = task.create_group(f"batch_{batch_idx:06d}")
                group.attrs["batch_size"] = int(out["preds"].shape[0])
                value = export.get("embedding_pooling_weights")
                if torch.is_tensor(value):
                    group.create_dataset("embedding_pooling_weights", data=value.detach().float().cpu().numpy())
                blocks = export.get("attention_weights_full", [])
                for j, block in enumerate(blocks):
                    value = block.get("pooling") if isinstance(block, dict) else None
                    if torch.is_tensor(value):
                        group.create_dataset(f"gfi_{j}_pooling_weights", data=value.detach().float().cpu().numpy())
    # Exclusivity is checked before writing; existing historical paths are never reused.
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        pd.DataFrame(data).to_csv(handle, index=False)
    print("Independent export: two-column predictions and task-indexed pooling; no shared-map HDF5. Scale is unchanged from model/datamodule.")
    return csv_path, h5_path
