from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import pandas as pd 

HERE     = Path(__file__).parent         
DATA_DIR = HERE.parent / "data"          
ART_DIR  = HERE / "artifacts"            
ART_DIR.mkdir(exist_ok=True)

SAMPLE_RATIO   = 0.20   # fraction of common clients to keep
TOP_K          = 500    # number of extra features to select
RARE_THRESHOLD = 0.01   # target is rare if positive rate < 1 %
MIN_POSITIVES  = 20     # skip a target if fewer than this many positives in sample
SEEDS          = [42, 314]

LGBM_PARAMS = dict(
    objective="binary", metric="auc",
    learning_rate=0.10, num_leaves=31,
    feature_fraction=0.8, n_jobs=-1,
    verbose=-1,
)


def build_sample_idx(target_df: pl.DataFrame, target_cols: list[str]) -> np.ndarray:
    n = len(target_df)
    counts = {c: int(target_df[c].sum()) for c in target_cols}
    rare_targets = [c for c, cnt in counts.items() if cnt < n * RARE_THRESHOLD]

    has_rare = target_df.select(
        pl.any_horizontal([pl.col(c) > 0 for c in rare_targets]).alias("r")
    )["r"].to_numpy()

    rare_idx   = np.where(has_rare)[0]
    common_idx = np.where(~has_rare)[0]

    rng = np.random.default_rng(SEEDS[0])
    sampled = rng.choice(common_idx, size=int(len(common_idx) * SAMPLE_RATIO), replace=False)
    idx = np.sort(np.concatenate([rare_idx, sampled]))

    print(f"Sample: {len(idx):,} rows ({len(idx) / n:.1%} of full train) "
          f"| rare clients: {len(rare_idx):,}")
    return idx


def aggregate_importances(
    extra_sub: "pd.DataFrame",
    target_df: pl.DataFrame,
    target_cols: list[str],
    idx: np.ndarray,
) -> np.ndarray:
    n_feats = extra_sub.shape[1]
    agg_gain = np.zeros(n_feats, dtype=np.float64)
    targets_used = 0

    for t in target_cols:
        y_t = target_df[t].to_numpy()[idx]
        n_pos = int(y_t.sum())
        n_neg = len(y_t) - n_pos

        if n_pos < MIN_POSITIVES:
            continue  

        spw = min(n_neg / n_pos, 50.0)  

        for seed in SEEDS:
            params = {**LGBM_PARAMS, "seed": seed, "scale_pos_weight": spw}
            model = lgb.train(
                params,
                lgb.Dataset(extra_sub, label=y_t),
                num_boost_round=80,
            )
            agg_gain += model.feature_importance(importance_type="gain")

        targets_used += 1

    print(f"Aggregated importances from {targets_used}/{len(target_cols)} targets "
          f"x {len(SEEDS)} seeds = {targets_used * len(SEEDS)} models")
    return agg_gain


def main() -> None:
    target_df   = pl.read_parquet(DATA_DIR / "train_target.parquet")
    target_cols = [c for c in target_df.columns if c.startswith("target_")]

    idx = build_sample_idx(target_df, target_cols)

    print("Loading extra features for sampled rows...")
    extra = pl.read_parquet(DATA_DIR / "train_extra_features.parquet")
    extra_sub = extra[idx.tolist()].drop("customer_id").to_pandas()
    del extra

    agg_gain = aggregate_importances(extra_sub, target_df, target_cols, idx)

    names = extra_sub.columns.tolist()
    order = np.argsort(agg_gain)[::-1]
    top_features = [names[i] for i in order[:TOP_K]]

    out = ART_DIR / "selected_features.pkl"
    joblib.dump(top_features, out)
    print(f"Saved top-{TOP_K} features -> {out}")


if __name__ == "__main__":
    main()
