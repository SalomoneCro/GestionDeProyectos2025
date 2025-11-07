from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Ensure 'src' is on sys.path when running as a script
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tuning.model_specs import MODELS
from tuning.optuna_objective import make_objective


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Optuna studies for multiple models (recall_weighted)"
    )
    ap.add_argument(
        "--models", type=str, default="all", help="Comma-separated model keys or 'all'"
    )
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument(
        "--timeout", type=int, default=900, help="Timeout per study in seconds"
    )
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--study-suffix",
        type=str,
        default="",
        help="Optional suffix for study name (e.g., _v2)",
    )
    ap.add_argument(
        "--jobs", type=int, default=1, help="Parallel trials (n_jobs) at Optuna level"
    )
    return ap.parse_args()


def _load_dataset(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def main() -> None:
    args = parse_args()

    # Hardcoded storage / dataset / target
    storage_path = Path("reports/tuning/optuna_studies.db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path.resolve()}"

    dataset_path = Path("data/processed/train_ready.parquet")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    X, y = _load_dataset(dataset_path, "Risk_Level")

    if args.models == "all":
        model_keys: List[str] = list(MODELS.keys())
    else:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in model_keys:
            if m not in MODELS:
                raise ValueError(
                    f"Unknown model key: {m}. Available: {sorted(MODELS.keys())}"
                )

    rows = []
    out_dir = Path("reports/tuning")
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in model_keys:
        study_name = f"tune_{key}{args.study_suffix}"
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=args.random_state),
            pruner=MedianPruner(),
            storage=storage_url,
            study_name=study_name,
            load_if_exists=True,
        )

        objective = make_objective(
            X=X,
            y=y,
            model_key=key,
            n_splits=args.n_splits,
            random_state=args.random_state,
            scoring="recall_weighted",
        )

        study.optimize(
            objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=args.jobs
        )

        best = study.best_trial
        row = {
            "study": study_name,
            "model": key,
            "best_recall_weighted": best.value,
            "recall_std": best.user_attrs.get("recall_std"),
            "params": best.params,
            "n_trials": len(study.trials),
        }
        rows.append(row)

        with open(
            out_dir / f"best_params_{study_name}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
