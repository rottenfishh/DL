from __future__ import annotations

import gc
from pathlib import Path

import category_encoders as ce
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from catboost import CatBoostClassifier

HERE            = Path(__file__).parent
DATA_DIR        = HERE.parent / "data"
ART_DIR         = HERE / "artifacts"
SUBMIT_FILE     = ART_DIR / "submission_improved_v2.parquet"
CHECKPOINT_FILE = ART_DIR / "checkpoint.pkl"

W_LGBM = 0.70
W_CAT  = 0.30
SEEDS  = [42, 777, 555, 2567, 429]

LGBM_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=128, min_data_in_leaf=200,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=5,
    n_jobs=-1, verbose=-1,
)
CAT_PARAMS = dict(
    iterations=250, learning_rate=0.1, depth=6,
    loss_function="Logloss", eval_metric="AUC",
    verbose=False, allow_writing_files=False,
)


def compute_aggs(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    cols = df.columns
    return df.select([
        pl.sum_horizontal(cols).alias(f"{prefix}_sum"),
        pl.mean_horizontal(cols).alias(f"{prefix}_mean"),
        pl.sum_horizontal([pl.col(c) != 0 for c in cols]).alias(f"{prefix}_nonzero"),
    ])


def build_feature_matrix(split: str, top_feats: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    main      = pl.read_parquet(DATA_DIR / f"{split}_main_features.parquet")
    extra_top = pl.read_parquet(DATA_DIR / f"{split}_extra_features.parquet", columns=top_feats)

    # Global aggs: load all extra, compute 2 numbers, free the big matrix
    extra_full  = pl.read_parquet(DATA_DIR / f"{split}_extra_features.parquet").drop("customer_id")
    aggs_global = compute_aggs(extra_full, "global")
    del extra_full

    aggs_local = compute_aggs(extra_top, "local")
    ids = main["customer_id"].to_pandas()

    X = pl.concat(
        [main.drop("customer_id"), extra_top, aggs_local, aggs_global],
        how="horizontal",
    ).to_pandas()

    return X, ids


def compute_spw(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return 1.0
    return float(min(n_neg / n_pos, 50.0))


def main() -> None:
    top_feats = joblib.load(ART_DIR / "selected_features.pkl")

    print("Building feature matrices...")
    X_tr, train_ids = build_feature_matrix("train", top_feats)
    X_te, test_ids  = build_feature_matrix("test",  top_feats)

    print("Appending level-1 meta-features...")
    meta_tr = pl.read_parquet(ART_DIR / "meta_train.parquet").to_pandas()
    meta_te = pl.read_parquet(ART_DIR / "meta_test.parquet").to_pandas()
    X_tr = pd.concat([X_tr, meta_tr], axis=1)
    X_te = pd.concat([X_te, meta_te], axis=1)

    target_df   = pd.read_parquet(DATA_DIR / "train_target.parquet")
    target_cols = [c for c in target_df.columns if c.startswith("target_")]
    y_global    = target_df[target_cols].mean(axis=1).values

	# change categorical columns to mean of all 41 targets. something like how many products this client has
    cat_cols = [c for c in X_tr.columns if c.startswith("cat_")]
    X_tr[cat_cols] = X_tr[cat_cols].astype(str)
    X_te[cat_cols] = X_te[cat_cols].astype(str)
    enc = ce.TargetEncoder(cols=cat_cols, smoothing=20)
    X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols], y_global).astype(np.float32)
    X_te[cat_cols] = enc.transform(X_te[cat_cols]).astype(np.float32)

    f64 = X_tr.select_dtypes("float64").columns
    X_tr[f64] = X_tr[f64].astype(np.float32)
    X_te[f64] = X_te[f64].astype(np.float32)

    print(f"Feature matrix: train {X_tr.shape}, test {X_te.shape}")

    preds: dict[str, np.ndarray] = {}
    if CHECKPOINT_FILE.exists():
        preds = joblib.load(CHECKPOINT_FILE)
        print(f"Resuming: {len(preds)}/{len(target_cols)} targets done")

    eps = 1e-6

    for i, col in enumerate(target_cols):
        pred_col = col.replace("target", "predict")
        if pred_col in preds:
            continue

        y = target_df[col].values
        if len(np.unique(y)) < 2:
            preds[pred_col] = np.full(len(X_te), -10.0, dtype=np.float32)
            continue

        spw = compute_spw(y)
        print(f"[{i + 1}/{len(target_cols)}] {col}  spw={spw:.1f}  seeds={SEEDS}")

        own_meta = f"meta_{col}"
        Xtr = X_tr.drop(columns=[own_meta], errors="ignore")
        Xte = X_te.drop(columns=[own_meta], errors="ignore")

        lgbm_pred = np.zeros(len(X_te), dtype=np.float64)
        cat_pred  = np.zeros(len(X_te), dtype=np.float64)

        for s in SEEDS:
            m_lgb = lgb.train(
                {**LGBM_PARAMS, "seed": s, "scale_pos_weight": spw},
                lgb.Dataset(Xtr, label=y),
                num_boost_round=250,
            )
            lgbm_pred += m_lgb.predict(Xte) / len(SEEDS)

            m_cat = CatBoostClassifier(
                **{**CAT_PARAMS, "random_seed": s, "class_weights": [1.0, spw]}
            )
            m_cat.fit(Xtr, y, silent=True)
            cat_pred += m_cat.predict_proba(Xte)[:, 1] / len(SEEDS)

        prob = lgbm_pred * W_LGBM + cat_pred * W_CAT
        prob = np.clip(prob, eps, 1 - eps)
        preds[pred_col] = np.log(prob / (1 - prob)).astype(np.float32)

        joblib.dump(preds, CHECKPOINT_FILE)
        gc.collect()

    sub = pd.read_parquet(DATA_DIR / "sample_submit.parquet", columns=["customer_id"])
    sub = sub.merge(
        pd.DataFrame({"customer_id": test_ids, **preds}),
        on="customer_id", how="left",
    ).fillna(-10.0)
    sub.to_parquet(SUBMIT_FILE, index=False)

    if len(preds) == len(target_cols):
        CHECKPOINT_FILE.unlink(missing_ok=True)

    print(f"\nSubmission saved -> {SUBMIT_FILE}")


if __name__ == "__main__":
    main()
