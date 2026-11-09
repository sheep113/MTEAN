# Modified for METAN portable release 2026-11-09.
from copy import deepcopy


def resolve_context_config(model_config):
    """
    Resolve optional context inputs.

    New format:
    context_inputs:
      sources:
        pca:
          enabled: true
          input_dim: 5
          ...
        genomic_relationship:
          enabled: false
          input_dim: 16
          ...
      fusion:
        late_fusion: true
        conditioning:
          enabled: true
          type: film
          target: post_embedding
          hidden_dim: 32
          init_identity: true

    Backward compatibility:
    old pca_covariates config is automatically mapped to
    a single PCA source with late fusion only.
    """

    context_cfg = model_config.get("context_inputs", None)

    legacy_mode = False

    # ========================================================
    # New context_inputs format
    # ========================================================

    if isinstance(context_cfg, dict):

        sources_cfg = context_cfg.get("sources", {})
        fusion_cfg = context_cfg.get("fusion", {})

        if not isinstance(sources_cfg, dict):
            raise ValueError(
                "context_inputs.sources must be a dict"
            )

        sources = {}

        for name, cfg in sources_cfg.items():

            if not isinstance(cfg, dict):
                raise ValueError(
                    f"context source '{name}' must be a dict"
                )

            cfg = deepcopy(cfg)

            enabled = bool(
                cfg.get("enabled", False)
            )

            if not enabled:
                continue

            input_dim = int(
                cfg.get("input_dim", 0)
            )

            if input_dim <= 0:
                raise ValueError(
                    f"context source '{name}' has "
                    f"invalid input_dim={input_dim}"
                )

            cfg["enabled"] = True
            cfg["input_dim"] = input_dim

            columns = cfg.get("columns", None)

            if columns is not None:
                if len(columns) != input_dim:
                    raise ValueError(
                        f"context source '{name}': "
                        f"len(columns)={len(columns)} "
                        f"!= input_dim={input_dim}"
                    )

            sources[name] = cfg

        conditioning = deepcopy(
            fusion_cfg.get(
                "conditioning",
                {}
            )
        )

        conditioning.setdefault(
            "enabled",
            False
        )

        conditioning.setdefault(
            "type",
            "film"
        )

        conditioning.setdefault(
            "target",
            "post_embedding"
        )

        conditioning.setdefault(
            "hidden_dim",
            32
        )

        conditioning.setdefault(
            "init_identity",
            True
        )

        late_fusion = bool(
            fusion_cfg.get(
                "late_fusion",
                True
            )
        )

    # ========================================================
    # Legacy PCA config
    # ========================================================

    else:

        legacy_mode = True

        pca_cfg = deepcopy(
            model_config.get(
                "pca_covariates",
                {}
            )
        )

        sources = {}

        if bool(
            pca_cfg.get(
                "enabled",
                False
            )
        ):

            n_components = int(
                pca_cfg.get(
                    "n_components",
                    5
                )
            )

            sources["pca"] = {
                "enabled": True,
                "input_dim": n_components,
                "scores_root": pca_cfg.get(
                    "scores_root",
                    "data/pca"
                ),
                "filename": "pca_scores.csv",
                "columns": [
                    f"PC{i}"
                    for i in range(
                        1,
                        n_components + 1
                    )
                ]
            }

        # IMPORTANT:
        # old PCA checkpoints must preserve old behaviour
        late_fusion = True

        conditioning = {
            "enabled": False,
            "type": "film",
            "target": "post_embedding",
            "hidden_dim": 32,
            "init_identity": True
        }

    total_dim = sum(
        int(cfg["input_dim"])
        for cfg in sources.values()
    )

    if (
        total_dim > 0
        and not late_fusion
        and not conditioning.get(
            "enabled",
            False
        )
    ):
        raise ValueError(
            "Context sources are enabled, but both "
            "late_fusion and conditioning are disabled."
        )

    if conditioning.get(
        "enabled",
        False
    ):

        ctype = str(
            conditioning.get(
                "type",
                "film"
            )
        ).lower()

        target = str(
            conditioning.get(
                "target",
                "post_embedding"
            )
        ).lower()

        if ctype != "film":
            raise ValueError(
                f"Unsupported context conditioning type: {ctype}"
            )

        if target != "post_embedding":
            raise ValueError(
                f"Unsupported conditioning target: {target}"
            )

        if total_dim <= 0:
            raise ValueError(
                "Context conditioning enabled "
                "but no context source is enabled."
            )

    return {
        "sources": sources,
        "total_dim": total_dim,
        "late_fusion": late_fusion,
        "conditioning": conditioning,
        "legacy_mode": legacy_mode
    }
