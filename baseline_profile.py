import pandas as pd
import numpy as np
import config

df = pd.read_csv(config.FULL_LOG_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow"] = df["timestamp"].dt.dayofweek

def build_entity_profile(g):
    known_resources = set(g["resource_accessed"].iloc[:max(5, len(g) // 3)])
    known_ips = set(g["source_ip"].iloc[:max(5, len(g) // 3)])
    known_geo = set(g["geo_location"].iloc[:max(5, len(g) // 3)])
    known_fp = g["device_fingerprint"].mode().iloc[0]
    known_auth = g["auth_method"].mode().iloc[0]
    return pd.Series({
        "hour_mean": g["hour"].mean(),
        "hour_std": max(g["hour"].std(), 0.5),
        "duration_mean": g["session_duration_min"].mean(),
        "duration_std": max(g["session_duration_min"].std(), 1.0),
        "known_resources": known_resources,
        "known_ips": known_ips,
        "known_geo": known_geo,
        "known_fp": known_fp,
        "known_auth": known_auth,
    })

profiles = df.groupby("entity_id").apply(build_entity_profile, include_groups=False)

def score_row(row):
    p = profiles.loc[row["entity_id"]]
    hour_z = abs(row["hour"] - p["hour_mean"]) / p["hour_std"]
    dur_z = abs(row["session_duration_min"] - p["duration_mean"]) / p["duration_std"]
    new_resource = 1.0 if row["resource_accessed"] not in p["known_resources"] else 0.0
    new_ip = 1.0 if row["source_ip"] not in p["known_ips"] else 0.0
    new_geo = 1.0 if row["geo_location"] not in p["known_geo"] else 0.0
    fp_mismatch = 1.0 if row["device_fingerprint"] != p["known_fp"] else 0.0
    auth_mismatch = 1.0 if row["auth_method"] != p["known_auth"] else 0.0
    auth_fail = 1.0 if not row["auth_success"] else 0.0
    return pd.Series({
        "hour_z": min(hour_z, 5),
        "dur_z": min(dur_z, 5),
        "new_resource": new_resource,
        "new_ip": new_ip,
        "new_geo": new_geo,
        "fp_mismatch": fp_mismatch,
        "auth_mismatch": auth_mismatch,
        "auth_fail": auth_fail,
    })

feat_cols = df.apply(score_row, axis=1)
df = pd.concat([df, feat_cols], axis=1)

df["baseline_score"] = (
    0.15 * df["hour_z"] +
    0.10 * df["dur_z"] +
    0.20 * df["new_resource"] +
    0.15 * df["new_ip"] +
    0.20 * df["new_geo"] +
    0.10 * df["fp_mismatch"] +
    0.05 * df["auth_mismatch"] +
    0.15 * df["auth_fail"]
)

threshold = df["baseline_score"].quantile(0.98)
df["baseline_flag"] = df["baseline_score"] >= threshold

print("Score threshold (top 2%):", round(threshold, 3))
print()
print(df.groupby("label")["baseline_score"].agg(["mean", "count"]).sort_values("mean", ascending=False))
print()
print("Recall by attack type at top-2% flag rate:")
print(df.groupby("label")["baseline_flag"].mean().sort_values(ascending=False))

df[["session_id", "entity_id", "label", "baseline_score", "baseline_flag"]].to_csv(
    config.BASELINE_SCORES_PATH, index=False
)
df.to_csv(config.FEATURES_PATH, index=False)