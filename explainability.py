import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

bundle = joblib.load(config.CLASSIFIER_MODEL_PATH)
clf = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_columns = bundle["feature_columns"]

import json
booster = clf.get_booster()
booster_config = json.loads(booster.save_config())
booster_config["learner"]["learner_model_param"]["base_score"] = "0.5"
booster.load_config(json.dumps(booster_config))

df = pd.read_csv(config.FEATURES_PATH)
df["auth_success"] = df["auth_success"].astype(float)
entity_dummies = pd.get_dummies(df["entity_type"], prefix="entity_type")

RAW_FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow", "session_duration_min", "auth_success",
    "hour_z", "dur_z", "new_resource", "new_ip", "new_geo",
    "fp_mismatch", "auth_mismatch", "auth_fail", "baseline_score",
]
X_all = pd.concat([df[RAW_FEATURE_COLS], entity_dummies], axis=1)
X_all = X_all.reindex(columns=feature_columns, fill_value=0)

classified = pd.read_csv(config.CLASSIFIED_ALERTS_PATH)

FEATURE_LABELS = {
    "hour_z": "unusual login hour for this entity",
    "dur_z": "unusual session duration for this entity",
    "new_resource": "accessed a resource never touched before",
    "new_ip": "connection from an unfamiliar IP",
    "new_geo": "geo-velocity: login from an unfamiliar location",
    "fp_mismatch": "device fingerprint mismatch (OS/MAC/protocol)",
    "auth_mismatch": "unusual authentication method for this entity",
    "auth_fail": "authentication failure",
    "baseline_score": "elevated deviation from historical baseline",
    "hour_sin": "atypical time-of-day pattern",
    "hour_cos": "atypical time-of-day pattern",
    "dow": "atypical day-of-week pattern",
    "session_duration_min": "session duration",
    "auth_success": "authentication outcome",
    "entity_type_user": "entity type: user",
    "entity_type_service_account": "entity type: service account",
    "entity_type_edge_device": "entity type: edge device",
}

flagged_idx = df.index[df["session_id"].isin(classified["session_id"])]
X_flagged = X_all.loc[flagged_idx]
session_ids_flagged = df.loc[flagged_idx, "session_id"].values

explainer = shap.TreeExplainer(clf)
shap_exp = explainer(X_flagged)
shap_values = shap_exp.values

print("SHAP values shape:", shap_values.shape)

class_index_lookup = classified.set_index("session_id")["predicted_attack_type"].map(
    lambda l: list(label_encoder.classes_).index(l)
)

reasons = []
for i, sid in enumerate(session_ids_flagged):
    pred_class_idx = class_index_lookup.get(sid, 0)
    if shap_values.ndim == 3:
        row_shap = shap_values[i, :, pred_class_idx]
    else:
        row_shap = shap_values[i]
    top_idx = np.argsort(-np.abs(row_shap))[:3]
    top_feats = [(feature_columns[j], row_shap[j]) for j in top_idx if row_shap[j] > 0]
    if not top_feats:
        top_feats = [(feature_columns[top_idx[0]], row_shap[top_idx[0]])]
    readable = [FEATURE_LABELS.get(f, f) for f, _ in top_feats]
    reasons.append("; ".join(readable))

explain_df = pd.DataFrame({
    "session_id": session_ids_flagged,
    "top_reasons": reasons,
})

result = classified.merge(explain_df, on="session_id", how="left")

def build_explanation_text(row):
    if row["predicted_attack_type"] == "normal":
        return f"Likely false positive (predicted normal, confidence {row['confidence']:.2f}) - {row['top_reasons']}"
    return f"Flagged as {row['predicted_attack_type']} (confidence {row['confidence']:.2f}) due to: {row['top_reasons']}"

result["explanation"] = result.apply(build_explanation_text, axis=1)

print()
print("Sample explanations:")
for _, r in result.sample(min(8, len(result)), random_state=config.RANDOM_SEED).iterrows():
    print(f"  [{r['label']:>20}] {r['explanation']}")

mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2)) if shap_values.ndim == 3 else np.mean(np.abs(shap_values), axis=0)
importance_df = pd.DataFrame({
    "feature": feature_columns,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False)

print()
print("Global feature importance (mean |SHAP| across flagged alerts):")
print(importance_df.to_string(index=False))

plt.figure(figsize=(8, 6))
plt.barh(importance_df["feature"][:12][::-1], importance_df["mean_abs_shap"][:12][::-1], color="#4C9AFF")
plt.xlabel("Mean |SHAP value|")
plt.title("Global feature importance across flagged alerts")
plt.tight_layout()
import os
os.makedirs(f"{config.BASE_DIR}/reports", exist_ok=True)
plt.savefig(f"{config.BASE_DIR}/reports/feature_importance.png", dpi=150)
print()
print("Saved feature importance chart to reports/feature_importance.png")

result.to_csv(config.EXPLAINED_ALERTS_PATH, index=False)