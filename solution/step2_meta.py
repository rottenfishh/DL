from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import KFold

HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
ART_DIR  = HERE / "artifacts"

N_SPLITS = 5
SEED     = 42

LGBM_BASE = dict(
    objective="binary", metric="auc",
    learning_rate=0.10, num_leaves=15, max_depth=4,
    n_estimators=75, n_jobs=-1, verbose=-1, seed=SEED,
)


def load_features(split: str, top_feats: list[str]) -> pd.DataFrame:
    main  = pl.read_parquet(DATA_DIR / f"{split}_main_features.parquet").drop("customer_id")
    extra = pl.read_parquet(DATA_DIR / f"{split}_extra_features.parquet", columns=top_feats)
    X = pl.concat([main, extra], how="horizontal").to_pandas()

    cat_cols = [c for c in X.columns if c.startswith("cat_")]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c].astype(str)).codes.astype(np.int16)

    return X.astype


def compute_spw(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return float(min(n_neg / n_pos, 50.0))


def main() -> None:
    top_feats = joblib.load(ART_DIR / "selected_features.pkl")

    print("Loading features...")
    X_tr = load_features("train", top_feats)
    X_te = load_features("test",  top_feats)

    target_df   = pd.read_parquet(DATA_DIR / "train_target.parquet")
    target_cols = [c for c in target_df.columns if c.startswith("target_")]

    n_targets  = len(target_cols)
    meta_train = np.zeros((len(X_tr), n_targets), dtype=np.float32)
    meta_test  = np.zeros((len(X_te), n_targets), dtype=np.float32)

    y_strat = (target_df[target_cols].sum(axis=1) > 0).astype(int).values
    skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_strat), 1):
        print(f"\nFold {fold}/{N_SPLITS}")
        X_f = X_tr.iloc[tr_idx]
        X_v = X_tr.iloc[va_idx]

        for i, t in enumerate(target_cols):
            y      = target_df[t].values
            y_fold = y[tr_idx]

            if len(np.unique(y_fold)) < 2:
                continue 

            spw    = compute_spw(y_fold)
            params = {**LGBM_BASE, "scale_pos_weight": spw}

            m = lgb.LGBMClassifier(**params)
            m.fit(X_f, y_fold)
            meta_train[va_idx, i] += m.predict_proba(X_v)[:, 1]
            meta_test[:, i]       += m.predict_proba(X_te)[:, 1] / N_SPLITS

    meta_cols = [f"meta_{t}" for t in target_cols]
    pl.DataFrame(dict(zip(meta_cols, meta_train.T))).write_parquet(ART_DIR / "meta_train.parquet")
    pl.DataFrame(dict(zip(meta_cols, meta_test.T ))).write_parquet(ART_DIR / "meta_test.parquet")
    print("\nSaved meta_train.parquet and meta_test.parquet")


if __name__ == "__main__":
    main()
