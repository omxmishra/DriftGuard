import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import config

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

df = pd.read_csv(config.FEATURES_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

df["dow_norm"] = df["dow"] / 6.0
df["duration_log"] = np.log1p(df["session_duration_min"])
df["auth_success_num"] = df["auth_success"].astype(float)

FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_norm", "duration_log", "auth_success_num",
    "hour_z", "dur_z", "new_resource", "new_ip", "new_geo",
    "fp_mismatch", "auth_mismatch", "auth_fail",
]

scaler = StandardScaler()
X_all = scaler.fit_transform(df[FEATURE_COLS].values)
df_scaled = df[["session_id", "entity_id", "label"]].copy()
for i, col in enumerate(FEATURE_COLS):
    df_scaled[col] = X_all[:, i]

WINDOW = config.WINDOW_SIZE
sequences, seq_session_ids, seq_entity_ids, seq_window_has_anomaly = [], [], [], []
cold_start_session_ids = []

for eid, g in df_scaled.groupby("entity_id"):
    arr = g[FEATURE_COLS].values
    sess_ids = g["session_id"].values
    labels = g["label"].values
    n = len(g)
    if n < WINDOW:
        cold_start_session_ids.extend(sess_ids.tolist())
        continue
    for start in range(0, n - WINDOW + 1):
        sequences.append(arr[start:start + WINDOW])
        seq_session_ids.append(sess_ids[start + WINDOW - 1])
        seq_entity_ids.append(eid)
        window_labels = labels[start:start + WINDOW]
        seq_window_has_anomaly.append(bool((window_labels != "normal").any()))

X_seq = np.stack(sequences).astype(np.float32)
window_has_anomaly = np.array(seq_window_has_anomaly)
print("Sequence windows built:", X_seq.shape)
print("Clean-normal windows used for training:", (~window_has_anomaly).sum())
print("Cold-start sessions (fallback to baseline score):", len(cold_start_session_ids))

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, window, hidden=32, latent=16):
        super().__init__()
        self.window = window
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden, latent)
        self.latent_to_dec = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output_layer = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
        z = self.enc_to_latent(h_n[-1])
        dec_input = self.latent_to_dec(z).unsqueeze(1).repeat(1, self.window, 1)
        dec_out, _ = self.decoder(dec_input)
        return self.output_layer(dec_out)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMAutoencoder(n_features=len(FEATURE_COLS), window=WINDOW).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_tensor = torch.tensor(X_seq)
train_tensor = X_tensor[~window_has_anomaly]
train_dataset = TensorDataset(train_tensor)
loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

EPOCHS = 15
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} - reconstruction MSE (normal windows only): {total_loss / len(train_dataset):.5f}")

model.eval()
with torch.no_grad():
    recon_all = model(X_tensor.to(device)).cpu().numpy()

per_window_error = np.mean((recon_all - X_seq) ** 2, axis=(1, 2))

seq_scores_df = pd.DataFrame({
    "session_id": seq_session_ids,
    "entity_id": seq_entity_ids,
    "sequence_score": per_window_error,
})

labels_lookup = df.set_index("session_id")["label"]
baseline_lookup = df.set_index("session_id")["baseline_score"]

seq_scores_df["label"] = seq_scores_df["session_id"].map(labels_lookup)
seq_scores_df["score_source"] = "sequence_model"

cold_df = pd.DataFrame({"session_id": cold_start_session_ids})
cold_df["entity_id"] = cold_df["session_id"].map(df.set_index("session_id")["entity_id"])
cold_df["label"] = cold_df["session_id"].map(labels_lookup)
cold_df["sequence_score"] = cold_df["session_id"].map(baseline_lookup)
cold_df["score_source"] = "baseline_fallback_cold_start"

final_df = pd.concat([seq_scores_df, cold_df], ignore_index=True)
final_df["baseline_score"] = final_df["session_id"].map(baseline_lookup)

final_df["seq_percentile"] = final_df["sequence_score"].rank(pct=True)
final_df["baseline_percentile"] = final_df["baseline_score"].rank(pct=True)
final_df["score_percentile"] = final_df[["seq_percentile", "baseline_percentile"]].max(axis=1)
final_df["flagged_by"] = np.where(
    final_df["seq_percentile"] >= final_df["baseline_percentile"], "sequence_model", "baseline_profile"
)
final_df["flag"] = final_df["score_percentile"] >= (1 - config.ALERT_BUDGET_PCT)

print()
print(f"Alert budget: top {config.ALERT_BUDGET_PCT*100:.1f}%")
print(final_df.groupby("label")["sequence_score"].agg(["mean", "count"]).sort_values("mean", ascending=False))
print()
print("Recall by attack type at alert budget:")
print(final_df.groupby("label")["flag"].mean().sort_values(ascending=False))
print()
print("Score source breakdown:")
print(final_df["score_source"].value_counts())

final_df.to_csv(config.SEQUENCE_SCORES_PATH, index=False)
torch.save(model.state_dict(), config.DETECTOR_MODEL_PATH)
joblib.dump(scaler, config.SCALER_PATH)