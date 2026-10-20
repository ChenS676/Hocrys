"""Aggregate results/results.csv -> mean +- std over seeds per (data, model, nu)."""
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/results.csv")
df["variant"] = df["model"] + df["nu"].fillna("").astype(str).str.replace(r"^(\d+)(\.0)?$", r"-nu\1", regex=True)
g = df.groupby(["data", "variant"]).agg(
    mae_y=("test_mae_y", "mean"), mae_y_sd=("test_mae_y", "std"),
    mae_f=("test_mae_f", "mean"), mae_f_sd=("test_mae_f", "std"),
    params=("params", "first"), train_s=("train_s", "mean"),
    infer_ms=("infer_ms_per_struct", "mean"), seeds=("seed", "nunique"))
pd.set_option("display.width", 200)
print(g.round(5).to_string())
