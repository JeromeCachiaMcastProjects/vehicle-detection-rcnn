import pandas as pd
from pathlib import Path

dfs = [pd.read_csv(f"results/fold_{i}_results.csv") for i in range(5)]
all_results = pd.concat(dfs)

best_per_fold = all_results.loc[all_results.groupby("fold")["mAP"].idxmax()].copy()

print("\nBest result per fold:")
print(best_per_fold[["fold","epoch","mAP","mAP_50","mAR","inference_ms"]].to_string(index=False))

summary = best_per_fold[["mAP","mAP_50","mAR","inference_ms"]].agg(["mean","std"])
print("\nCross-validation summary (mean ± std):")
print(summary)

all_results.to_csv("results/all_folds.csv", index=False)
best_per_fold.to_csv("results/best_per_fold.csv", index=False)
summary.to_csv("results/summary.csv")
print("\nAll saved to results/")