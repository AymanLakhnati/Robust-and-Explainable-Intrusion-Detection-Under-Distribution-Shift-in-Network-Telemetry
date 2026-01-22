from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.data.load import coerce_binary_label, load_csv
from src.data.preprocess import PreprocessArtifacts, build_preprocessor, infer_feature_columns, transform
from src.data.split import DomainSplit, filter_by_domains, make_domain_split
from src.utils.config import deep_get, ensure_dir, load_yaml, save_yaml
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess data and create domain-based splits.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    random_seed = int(deep_get(cfg, ["project", "random_seed"], 42))
    set_global_seed(random_seed)

    data_cfg: Dict[str, Any] = deep_get(cfg, ["data"], {})
    paths_cfg: Dict[str, Any] = deep_get(cfg, ["paths"], {})

    csv_path = data_cfg.get("csv_path")
    if not csv_path:
        raise ValueError("data.csv_path must be set in config.")

    label_col = data_cfg.get("label_col", "label")
    domain_col = data_cfg.get("domain_col", "day")

    test_last_n_domains = data_cfg.get("test_last_n_domains")
    train_domains = data_cfg.get("train_domains")
    test_domains = data_cfg.get("test_domains")

    recent_val_cfg: Dict[str, Any] = deep_get(cfg, ["data", "recent_val"], {})
    recent_val_n_domains = int(recent_val_cfg.get("n_domains", 1))

    preprocess_cfg: Dict[str, Any] = deep_get(cfg, ["preprocess"], {})
    drop_cols = preprocess_cfg.get("drop_cols", [])
    categorical_max_unique = int(preprocess_cfg.get("categorical_max_unique", 50))
    scale_numeric = bool(preprocess_cfg.get("scale_numeric", True))

    artifacts_dir = ensure_dir(paths_cfg.get("artifacts_dir", "artifacts"))

    df = load_csv(csv_path)
    if label_col not in df.columns:
        raise KeyError(f"label_col '{label_col}' not found in CSV.")
    if domain_col not in df.columns:
        raise KeyError(f"domain_col '{domain_col}' not found in CSV.")

    eval_cfg: Dict[str, Any] = deep_get(cfg, ["eval"], {})
    positive_label = eval_cfg.get("positive_label", 1)
    df[label_col] = coerce_binary_label(df[label_col], positive_label=positive_label)

    split = make_domain_split(
        df,
        domain_col=domain_col,
        test_last_n_domains=test_last_n_domains,
        train_domains=train_domains,
        test_domains=test_domains,
        recent_val_n_domains=recent_val_n_domains,
    )

    train_df = filter_by_domains(df, domain_col=domain_col, domains=split.train_domains)
    test_df = filter_by_domains(df, domain_col=domain_col, domains=split.test_domains)
    recent_val_df = filter_by_domains(df, domain_col=domain_col, domains=split.recent_val_domains)

    feature_cols = infer_feature_columns(
        df,
        label_col=label_col,
        domain_col=domain_col,
        drop_cols=drop_cols,
    )

    artifacts = build_preprocessor(
        train_df,
        feature_cols=feature_cols,
        categorical_max_unique=categorical_max_unique,
        scale_numeric=scale_numeric,
    )

    train_df.to_csv(artifacts_dir / "train.csv", index=False)
    test_df.to_csv(artifacts_dir / "test.csv", index=False)
    recent_val_df.to_csv(artifacts_dir / "recent_val.csv", index=False)

    meta = {
        "label_col": label_col,
        "domain_col": domain_col,
        "feature_cols": feature_cols,
        "domain_split": {
            "train_domains": split.train_domains,
            "test_domains": split.test_domains,
            "recent_val_domains": split.recent_val_domains,
        },
    }
    save_yaml(meta, artifacts_dir / "preprocess_meta.yaml")
    joblib.dump(artifacts.preprocessor, artifacts_dir / "preprocessor.joblib")

    save_yaml(cfg, artifacts_dir / "config_used.yaml")


if __name__ == "__main__":
    main()

