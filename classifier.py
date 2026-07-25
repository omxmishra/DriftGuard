import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import config

df = pd.read_csv(config.FEATURES_PATH)

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow", "session_duration_min", "auth_success",
    "hour_z", "dur_z", "new_resource", "new_ip", "new_geo",
    "fp_mismatch", "auth_mismatch", "auth_fail", "baseline_score",
]

df["auth_success"] = df["auth_success"].astype(float)
entity_dummies = pd.get_dummies(df["entity_type"], prefix="entity_type")
X_all = pd.concat([df[FEATURE_COLS], entity_dummies], axis=1)

anomaly_mask = df["label"] != "normal"
normal_sample = df[~anomaly_mask].sample(n=min(600, (~anomaly_mask).sum()), random_state=config.RANDOM_SEED)
train_pool_mask = anomaly_mask.copy()
train_pool_mask.loc[normal_sample.index] = True

X_pool = X_all[train_pool_mask].reset_index(drop=True)
y_pool = df.loc[train_pool_mask, "label"].reset_index(drop=True)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_pool)

X_train, X_test, y_train, y_test = train_test_split(
    X_pool, y_encoded, test_size=0.25, stratify=y_encoded, random_state=config.RANDOM_SEED
)

class_counts = pd.Series(y_train).value_counts()
sample_weight = pd.Series(y_train).map(lambda c: len(y_train) / (len(class_counts) * class_counts[c])).values

clf = XGBClassifier(
    objective="multi:softprob",
    num_class=len(label_encoder.classes_),
    eval_metric="mlogloss",
    max_depth=4,
    n_estimators=200,
    learning_rate=0.1,
    base_score=0.5,
    random_state=config.RANDOM_SEED,
)
clf.fit(X_train, y_train, sample_weight=sample_weight)

y_pred = clf.predict(X_test)
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "class"})
report_df = report_df[report_df["class"].isin(label_encoder.classes_)]
report_df[["precision", "recall", "f1-score"]] = report_df[["precision", "recall", "f1-score"]].round(3)
report_df.to_csv(config.CLASSIFIER_METRICS_PATH, index=False)

print("Classification report (attack-type prediction, held-out anomalies):")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

print("Confusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index=label_encoder.classes_,
    columns=label_encoder.classes_,
))

seq_scores = pd.read_csv(config.SEQUENCE_SCORES_PATH)
flagged = seq_scores[seq_scores["flag"] == True].copy()

flagged_features = X_all.loc[df["session_id"].isin(flagged["session_id"])].copy()
flagged_session_ids = df.loc[df["session_id"].isin(flagged["session_id"]), "session_id"].values
flagged_features = flagged_features.reindex(columns=X_pool.columns, fill_value=0)

probs = clf.predict_proba(flagged_features)
pred_idx = np.argmax(probs, axis=1)
pred_labels = label_encoder.inverse_transform(pred_idx)
confidence = np.max(probs, axis=1)

classified = pd.DataFrame({
    "session_id": flagged_session_ids,
    "predicted_attack_type": pred_labels,
    "confidence": confidence,
})
classified = classified.merge(flagged, on="session_id", how="left")

print()
print("Predicted attack-type distribution among flagged alerts:")
print(classified["predicted_attack_type"].value_counts())
print()
print("Agreement rate (predicted == true label, where true label known):")
print((classified["predicted_attack_type"] == classified["label"]).mean())

classified.to_csv(config.CLASSIFIED_ALERTS_PATH, index=False)
joblib.dump({"model": clf, "label_encoder": label_encoder, "feature_columns": list(X_pool.columns)}, config.CLASSIFIER_MODEL_PATH)