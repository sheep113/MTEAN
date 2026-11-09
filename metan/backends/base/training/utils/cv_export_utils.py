# Modified for METAN portable release 2026-11-09.
"""Opt-in CV export policy; absent configuration preserves historical exports."""
def resolve_cv_export(config):
    value = config.get("cv_export", {})
    if not isinstance(value, dict):
        raise ValueError("cv_export must be a dictionary")
    enabled = value.get("enabled", True)
    pooling = value.get("save_pooling_weights", True)
    stages = value.get("stages", ["train", "val"])
    if not isinstance(enabled, bool) or not isinstance(pooling, bool):
        raise ValueError("cv_export enabled/save_pooling_weights must be boolean")
    if not isinstance(stages, list) or any(x not in ("train", "val") for x in stages) or len(set(stages)) != len(stages):
        raise ValueError("cv_export stages must be unique train/val names; Test is forbidden")
    return {"stages": list(stages) if enabled else [], "save_pooling_weights": pooling}


def save_predictions_csv_only(outputs, dataset, output_dir, stage, phenotype_names):
    """Export existing sequential predict_step outputs, preserving their original scale."""
    import torch
    import pandas as pd
    from pathlib import Path
    from torch.utils.data import Subset
    if stage not in ("train", "val"):
        raise ValueError("Compact CV export only permits train/val")
    if not outputs:
        raise ValueError("No prediction output")
    pred = torch.cat([x["preds"].detach().float().cpu() for x in outputs], dim=0)
    labels = torch.cat([x["labels"].detach().float().cpu() for x in outputs], dim=0)
    if pred.shape != labels.shape or pred.shape != (len(dataset), len(phenotype_names)):
        raise ValueError("Prediction/label/trait/sample shape mismatch")
    if not bool(torch.isfinite(pred).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError("Nonfinite predictions/labels")
    ordered = []
    for index in range(len(dataset)):
        base, local = dataset, index
        while isinstance(base, Subset):
            local = int(base.indices[local])
            base = base.dataset
        sid = base.sample_ids[local]
        ordered.append(sid.decode() if isinstance(sid, bytes) else str(sid))
    if len(ordered) != len(set(ordered)):
        raise ValueError("Duplicate export sample IDs")
    data = {"sample_id": ordered}
    for i, name in enumerate(phenotype_names):
        data[f"{name}_pred"] = pred[:, i].numpy()
        data[f"{name}_label"] = labels[:, i].numpy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stage}_predictions.csv"
    with path.open("x", newline="") as stream:
        pd.DataFrame(data).to_csv(stream, index=False)
    return path
